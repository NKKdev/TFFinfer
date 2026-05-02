//
// Created by nkk on 2025/10/21.
//

#include "InferRuntime.h"
#include "model/base/ModelDetectorRegistry.h"
#include "model/llama/LLAMACreator.h"
#include "model/base/ModelConfigReader.h"
#include "mem/BaseDefine.h"
#include "sampler/LLMSampler.h"
namespace tff::core::runtime {
    bool LLMInferRuntime::load_model(const std::vector<std::string> &model_files_path,
                                     const tff::core::model::ModelConfig &params) {
        tff::log::Logger::info("Loading model: {%s}", model_files_path[0].c_str());
        if (model_files_path.empty()) {
            return false;
        }
        bool bRet = true;
        auto model_detector = tff::core::model::ModelDetectorRegistry::find_dector(
            tff::utils::get_file_ext(model_files_path[0]));

        this->_model_loader = model_detector->create_loader();
        this->_model_loader->load_from_file(model_files_path, params);
        this->build_model_creator();
        this->_vocabulary_ptr = std::make_unique<tff::core::model::LLMVocabulary>();

        this->load_hparams(params._is_fuse_op, params._kv_data_type);
        this->load_vocab();
        this->build_layers();
        return bRet;
    }

    bool LLMInferRuntime::load_model_config(const std::string &model_config_file_path,
                                            tff::core::model::ModelConfig &params) {
        tff::core::model::ModelConfigReader::Config cfg = tff::core::model::ModelConfigReader::read(
            model_config_file_path);
        params._architectures = cfg.architectures;
        return true;
    }

    bool LLMInferRuntime::build_model_creator() {
        auto creator = tff::factory::ModuleFactory::instance()->create_shared<ModelCreatorBase>(
            MODEL_CREATOR_FLAG, std::string(this->_model_loader->get_arch_name()));
        if (creator == nullptr) {
            tff::log::Logger::error("Model creator is nullptr.");
            return false;
        }
        this->_model_creator = creator;
        auto &cfg = this->_model_loader->get_model_config();
        this->_model_creator->_graph_ctx._n_embd_head = cfg._n_embd / cfg._n_head_arr[0];
        this->_model_creator->_graph_ctx._max_seq_len = cfg._n_ctx;
        this->_model_creator->_graph_ctx._n_embd_head_k = cfg._n_embd_head_k;
        this->_model_creator->_graph_ctx._n_embd_head_v = cfg._n_embd_head_v;
        this->_model_creator->_graph_ctx._n_head = cfg._n_head_arr[0];
        this->_model_creator->_graph_ctx._n_head_kv = cfg._n_head_kv_arr[0];
        this->_model_creator->_graph_ctx._n_layer = cfg._n_layer;
        this->_model_creator->_graph_ctx._use_fp16 = cfg._use_f16;
        this->_model_creator->_graph_ctx._use_mmap = cfg._use_mmap;
        this->_model_creator->_graph_ctx._rope_freq_base = cfg._rope_freq_base;
        this->_model_creator->_graph_ctx._rope_freq_scale = cfg._rope_freq_scale;
        this->_model_creator->_graph_ctx._model_loader = this->_model_loader;
        this->_model_creator->_graph_ctx._f_norm_rms_eps = cfg._f_norm_rms_eps;
        this->_model_creator->_graph_ctx._rope_type = cfg._rope_type;
        this->_model_creator->_graph_ctx._is_fuse = cfg._is_fuse_op;
        this->_model_creator->_graph_ctx._n_output = cfg._n_output;

        return true;
    }

    bool LLMInferRuntime::init_device() {
        this->_device_manager = std::dynamic_pointer_cast<device::DeviceManager>(
            tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                DEVICE_MANAGER_FLAG,
                tff::factory::ModuleKeyType(DEVICE_MANAGER_FLAG)));
        if (this->_device_manager == nullptr) {
            return false;
        }
        return true;
    }

    bool LLMInferRuntime::init_runtime_context() {
        tff::log::Logger::info("Initializing runtime context.");
        bool ret = true;
        ret &= init_kvcache();
        ret &= load_tensor_data();
        return ret;
    }

    bool LLMInferRuntime::init_kvcache() {
        //
        LLMKVCache::KVConfig kv_cfg;
        kv_cfg._n_embd_head = this->_model_config._n_embd_head_k;
        kv_cfg._n_head = this->_model_config._n_head_arr[0];
        kv_cfg._n_head_kv = this->_model_config._n_head_kv_arr[0];
        kv_cfg._n_layer = this->_model_config._n_layer;
        kv_cfg._use_sliding_window = this->_model_config._n_swa != 0;
        kv_cfg._max_tokens = this->_model_config._n_ctx;
        kv_cfg._use_f16 = this->_model_config._use_f16;
        kv_cfg._data_type = this->_model_config._kv_data_type;
        for (auto &[device_id, device]: this->_device_manager->devices()) {
            if (!device) {
                tff::log::Logger::error("No valid device found in _devices.");
                return false;
            }
            //
            auto type_size = memory::type_traits_auto[kv_cfg._data_type]._type_size;

            const float one_page_size = kv_cfg._n_embd_head * kv_cfg._n_head_kv * PAGE_SIZE * type_size;
            tff::log::Logger::info("KV Cache: Size per page: {%lf} bytes", one_page_size);

            tff::log::Logger::info("KV Cache: Target devices: {%d}", device_id);

            auto model_ctx = this->_model_loader->get_model_ctx();

            size_t free_mem = 0;
            size_t total_mem = 0;
            device->get_device_mem(device_id, &free_mem, &total_mem);
            tff::log::Logger::info("Device {%d}: Total memory: {%lld} bytes, Free memory: {%lld} bytes",
                                   device_id, total_mem, free_mem);
            //

            //预留模型上下文权重和其他开销的显存;
            const size_t context_reserve = model_ctx->_max_tensor_byte_size * model_ctx->_tensor_info.size();
            free_mem -= context_reserve;
            tff::log::Logger::info("Reserved memory for model context and overhead: {%lld} bytes", context_reserve);

            if (free_mem <= 0) {
                tff::log::Logger::error(
                    "Insufficient GPU memory. After reservation, free memory is {%lf} bytes (<= 0).",
                    free_mem);
                return false;
            }

            //计算总页数并创建KV Cache
            kv_cfg._total_pages = min(static_cast<int>(free_mem / one_page_size),
                kv_cfg._n_layer * kv_cfg._max_tokens / PAGE_SIZE);
            tff::log::Logger::info("KV Cache: Total available free memory for KV: {%lld} bytes",
                                   (free_mem));
            tff::log::Logger::info("KV Cache: Total pages calculated: {%d} ({%lld} bytes per page)",
                                   kv_cfg._total_pages, static_cast<size_t>(one_page_size));

            if (kv_cfg._total_pages == 0) {
                tff::log::Logger::error(
                    "Calculated total KV cache pages is 0. Available memory ({%lld}) is less than one page size ({%lld}).",
                    static_cast<size_t>(free_mem), static_cast<size_t>(one_page_size));
                return false;
            }
            if (!this->_mem_manager_ptr->init(device_id)) {
                tff::log::Logger::error("device %d memory manager init failed!!", device_id);
                return false;
            }
            try {
                this->_kv_cache_ptr[device_id] = std::make_shared<LLMKVCache>(device_id,
                                                                              this->_model_config._kv_data_type,
                                                                              kv_cfg, this->_mem_manager_ptr);
                tff::log::Logger::info("KV Cache successfully initialized with {%d} pages.", kv_cfg._total_pages);
            } catch (const std::exception &e) {
                tff::log::Logger::error("Failed to create LLMKVCache instance. Exception: {%s}", e.what());
                return false;
            } catch (...) {
                tff::log::Logger::error("Failed to create LLMKVCache instance. Unknown exception occurred.");
                return false;
            }
            if (this->_mem_manager_ptr == nullptr) {
                tff::log::Logger::info("Memory Manager created failed.");
                return false;
            }
        }
        //

        return true; // 初始化成功
    }

    bool LLMInferRuntime::init_graph(std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
        if (this->_layer_map.empty()) {
            tff::log::Logger::error("model layer is invalid!!\n");
            return false;
        }
        tff::log::Logger::info("Initializing graph");

        this->_model_creator->_graph_ctx._kv_cache_ptr = this->_kv_cache_ptr;
        this->_model_creator->build_graph(this->_layer_map, graph_ptr);
        this->_graph_optimizer->optimize(this->_model_creator->_graph_ctx, graph_ptr);
        return true;
    }
    bool LLMInferRuntime::init_io_graph(std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
        if (this->_layer_map.empty()) {
            tff::log::Logger::error("model layer is invalid!!\n");
            return false;
        }
        tff::log::Logger::info("Initializing graph");

        this->_model_creator->_graph_ctx._kv_cache_ptr = this->_kv_cache_ptr;
        this->_model_creator->build_mem_graph(this->_layer_map, graph_ptr);
        this->_graph_optimizer->optimize(this->_model_creator->_graph_ctx, graph_ptr);
        return true;
    }
    void LLMInferRuntime::build_mem_offset(const std::shared_ptr<tff::core::runtime::LLMMemManager> &_mem_manager_ptr,
                                           const std::shared_ptr<graph::Graph> &graph_ptr,
                                           std::unordered_map<std::string,
                                               std::unordered_map<int, size_t> > &
                                           mem_buffer_offset_map) const {
        for (auto &node: graph_ptr->total_nodes()) {
            if (node->op_type() == core::graph::TffOpType::TFF_OP_MEM_REF ||
                node->op_type() == TFF_OP_VIEW || node->op_type() == TFF_OP_MAP2CPU) {
                continue;
            }

            const auto &device_id = node->get_tensor()->get_allocator()->_device_id;
            const auto &current_device =
                this->_device_manager->get_device(device_id);
            auto [start, end] = graph_ptr->get_lifetime(node);
            //tff::log::Logger::info("node： %s start: %d, end: %d", node->name().c_str(), start, end);

            auto mem_offset = _mem_manager_ptr->allocate_memory(node->get_tensor()->get_bytes(),
                                                                start, end, device_id, node->get_tensor()->memory_type());
            //node->get_tensor()->set_external_memory_index(mem_offset);
            auto device_type_flag = current_device->get_device_type_flag(device_id);
            // tff::log::Logger::info("node: %s mem start offset: %lld, mem end offset: %lld", node->name().c_str(),
            //                        mem_offset, mem_offset + node->get_tensor()->get_bytes());
            auto iter = mem_buffer_offset_map.find(device_type_flag);
            if (iter != mem_buffer_offset_map.end()) {
                auto device_iter = iter->second.find(device_id);
                if (device_iter != iter->second.end()) {
                    device_iter->second = device_iter->second<mem_offset
                                                                  ? mem_offset
                                                                  : device_iter->second;
                } else {
                    iter->second.insert(std::make_pair(device_id, mem_offset));
                }
            } else {
                std::unordered_map<int, size_t> device_mem_offset_map;
                device_mem_offset_map.insert(std::make_pair(device_id, mem_offset));
                mem_buffer_offset_map.insert(std::make_pair(device_type_flag, device_mem_offset_map));
            }
        }
    }

    int32_t LLMInferRuntime::sample_token() {
        static thread_local std::mt19937 rng{std::random_device{}()};
        float *logits = static_cast<float *>(this->_model_creator->_graph_ctx._logits->get_buffer()->ptr() +
                (this->_model_creator->_graph_ctx._logits->get_shape()[1] - this->_model_config._n_output) *
                this->_model_creator->_graph_ctx._logits->get_strides()[1]);

        return core::sampling::LLMSampler::sample(this->_model_creator->_graph_ctx._logits->get_shape()[0],
            logits,
            this->_model_config._sampling_config, rng);
    }

    bool LLMInferRuntime::init_mem_manager(const std::shared_ptr<graph::Graph> &graph_ptr,
                                           std::shared_ptr<tff::core::runtime::LLMMemManager> &_mem_manager_ptr) const {
        if (graph_ptr == nullptr) {
            tff::log::Logger::error("model graph is invalid!!\n");
            return false;
        }
        if (_mem_manager_ptr == nullptr) {
            tff::log::Logger::error("model memory manager is invalid!!\n");
            return false;
        }
        std::unordered_map<std::string, std::unordered_map<int, size_t> >
                mem_buffer_offset_map;
        this->build_mem_offset(_mem_manager_ptr, graph_ptr, mem_buffer_offset_map);

        bool bRet = true;
        bool is_cpu_init = false;
        for (auto &mem_buffer_map: mem_buffer_offset_map) {
            if (mem_buffer_map.first == "CPU") {
                is_cpu_init = true;
            }
            for (auto &mem_offset: mem_buffer_map.second) {
                bRet &= _mem_manager_ptr->init(mem_offset.first);
            }
        }
        if (!is_cpu_init) {
            bRet &= _mem_manager_ptr->init(-1); //cpu id = -1;
        }
        this->_model_creator->_graph_ctx._mem_manager_ptr = _mem_manager_ptr;
        return bRet;
    }

    bool LLMInferRuntime::infer(const int n_predict, std::vector<std::string> &generate_str_vec) {
        tff::log::Logger::info("infer start");
        bool bRet = true;
        for (auto &batch: this->_llm_batch_manager_ptr->_ubatches) {
            bRet &= this->prefill(batch);
            std::string generate_str;
            bRet &= this->decode(batch, n_predict, generate_str);
            generate_str_vec.push_back(generate_str);
        }

        return bRet;
    }

    bool LLMInferRuntime::prefill(std::shared_ptr<LLMBatch> &ubatch) {
        tff::log::Logger::info("prefill start");
        this->_model_creator->_graph_ctx._is_prefill = true;
        this->build_inputs(ubatch);
        this->build_output();
        if (!this->_prefill_graph_ptr) {
            this->init_graph(this->_prefill_graph_ptr);
        }

        try {
            this->_task_manager->build_task_schedule(schedule::TaskType::TFF_TASK_TYPE_INFER,
                                                     this->_prefill_graph_ptr);

            this->_task_manager->run(tff::schedule::TaskType::TFF_TASK_TYPE_INFER);
        } catch (const std::exception &e) {
            tff::log::Logger::error("Prefill forward failed: {%s}", e.what());
            return false;
        }

        //
        ubatch->_tokens.push_back(this->sample_token());

        ubatch->_n_tokens++;
        ubatch->_pos.push_back(ubatch->_tokens.size() - 1);
        ubatch->_token_seq_ids.push_back(ubatch->_token_seq_ids[ubatch->_token_seq_ids.size() - 1]);
        ubatch->_logits[ubatch->_logits.size() - 1] = 0;
        ubatch->_logits.push_back(1);
        tff::log::Logger::info("prefill token: %d", ubatch->_tokens.back());
        return true;
    }

    bool LLMInferRuntime::decode(std::shared_ptr<LLMBatch> &ubatch,
                                 const int &n_predict, std::string &generate_str) {
        tff::log::Logger::info("decode start");
        this->_model_creator->_graph_ctx._is_prefill = false;

        std::shared_ptr<LLMBatch> batch = std::make_shared<LLMBatch>();
        batch = ubatch;
        batch->_pos = {batch->_pos.back()};
        auto pos = batch->_pos.back();
        batch->_tokens = {batch->_tokens.back()};
        batch->_n_tokens = batch->_tokens.size();

        this->_model_creator->_graph_ctx._n_tokens = batch->_n_tokens;

        this->build_inputs(batch);
        this->build_output();
        std::vector<int64_t> generate_token;
        generate_token.reserve(n_predict);
        if (!this->_decode_graph_ptr) {
            this->init_graph(this->_decode_graph_ptr);
            this->_task_manager->build_task_schedule(schedule::TaskType::TFF_TASK_TYPE_INFER, this->_decode_graph_ptr);
        }
        time_t t1 = clock();
        for (int step = 0; step < n_predict; ++step) {
            generate_token.push_back(batch->_tokens.back());
            batch->_pos = {pos};
            batch->_tokens = {batch->_tokens.back()};
            batch->_n_tokens = batch->_tokens.size();
            this->_model_creator->_graph_ctx._n_tokens = batch->_n_tokens;

            this->build_inputs(batch);
            this->build_output();

            this->_task_manager->run(tff::schedule::TaskType::TFF_TASK_TYPE_INFER);

            batch->_tokens.push_back(this->sample_token());
            batch->_n_tokens++;
            pos++;
            batch->_token_seq_ids.push_back(ubatch->_token_seq_ids[ubatch->_token_seq_ids.size() - 1]);
            batch->_logits[ubatch->_logits.size() - 1] = 0;
            batch->_logits.push_back(1);

            update_kv_cache(this->_model_creator->_graph_ctx, batch);
            const auto &eog_tokens = this->_vocabulary_ptr->get_eog_tokens();
            std::string token_str = this->_vocabulary_ptr->_cache_token_to_piece[batch->_tokens.back()];
            generate_str += token_str;
            std::cout << token_str << std::flush;
            if (std::find(eog_tokens.begin(), eog_tokens.end(), batch->_tokens.back()) != eog_tokens.end()){
                break;
            }
        }
        std::cout<<std::endl;
        time_t t2 = clock();
        tff::log::Logger::info("%d token decode time: %d s", n_predict, (t2 - t1) / CLOCKS_PER_SEC);
        tff::log::Logger::info("generate string: %s", generate_str.c_str());
        return true;
    }

    int LLMInferRuntime::encode(const std::vector<std::string> &prompt_batches) {
        tff::log::Logger::info("encode prompt...");
        if (prompt_batches.empty()) {
            tff::log::Logger::error("Prompt is empty.");
            return -1;
        }

        const size_t batch_size = prompt_batches.size();
        std::unordered_map<int, std::string> seq_prompts;
        //std::vector<std::vector<int> > tokenized_batch;
        size_t max_seq_len = 0;
        for (int i = 0; i < batch_size; ++i) {
            const auto &batch = prompt_batches[i];
            std::vector<int> tokens;
            this->_vocabulary_ptr->tokenize(batch, tokens);

            if (tokens.size() > this->_model_config._n_ctx) {
                tff::log::Logger::warning("Prompt {%d} length ({%d}) exceeds context length ({%d}). Truncating.",
                                          i, tokens.size(), this->_model_config._n_ctx);
                tokens.resize(this->_model_config._n_ctx);
            }
            max_seq_len = std::max(max_seq_len, tokens.size());
            //tokenized_batch.push_back(std::move(tokens));
            seq_prompts[i] = batch;
            for (int n = 0; n < this->_model_config._n_layer; ++n) {
                for (auto &cache: this->_kv_cache_ptr | std::views::values) {
                    cache->build_layer_kvcache_context(i, n);
                }
            }
            tff::log::Logger::info("Batch {%d}: tokenized {%d} tokens.", i, tokens.size());
        }
        //batch manager init;
        if (!this->_llm_batch_manager_ptr->init(seq_prompts, this->_vocabulary_ptr, false,
                                                LLMBatchManager::BatchSplitType::TTF_BATCH_SPLIT_SEQ)) {
            tff::log::Logger::error("LLMInferRuntime batch init failed.");
            return -1;
        }

        //this->build_inputs();

        //todo set input_pos_tensor ;
        return max_seq_len;
    }

    void LLMInferRuntime::build_inputs(std::shared_ptr<LLMBatch> &batch) {
        //set embedding layer input
        this->_model_creator->_graph_ctx._seq_id = batch->_token_seq_ids[0];
        auto &tokens_data = batch->_tokens;
        auto &input_pos = batch->_pos;
        auto token_tensor = std::make_shared<tff::core::memory::Tensor>(
            tff::core::memory::DataType::TFF_DATA_TYPE_I32,
            memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
            std::array<int64_t, MAX_TENSOR_DIM>{
                static_cast<int64_t>(tokens_data.size()), 1, 1, 1
            }, true);
        token_tensor->set_buffer_data(tokens_data.data(),
                                      tokens_data.size() * memory::type_traits_auto[
                                          tff::core::memory::DataType::TFF_DATA_TYPE_I32]._type_size);
        token_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN);

        auto input_pos_tensor = std::make_shared<tff::core::memory::Tensor>(
            tff::core::memory::DataType::TFF_DATA_TYPE_I32,
            memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
            std::array<int64_t, MAX_TENSOR_DIM>{
                static_cast<int64_t>(input_pos.size()),
                1, 1, 1
            }, true);
        input_pos_tensor->set_buffer_data(input_pos.data(),
                                          input_pos.size() * memory::type_traits_auto[
                                              tff::core::memory::DataType::TFF_DATA_TYPE_I32]._type_size);
        input_pos_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_TOKEN_POS);

        auto input_layer_iter = this->_layer_map.find(LLM_TENSOR_LAYER_INPUT);
        if (input_layer_iter != this->_layer_map.end()) {
            for (auto &layers: input_layer_iter->second) {
                auto &layer = layers.second;
                auto input_pos_layer_iter = layer.find(memory::ModelTensorType::LLM_TENSOR_TOKEN_POS);
                if (input_pos_layer_iter == layer.end()) {
                    auto input_pos_layer = std::make_shared<tff::core::model::layer::ModelLayerObject>();
                    input_pos_layer->_type = LLM_TENSOR_LAYER_INPUT;
                    input_pos_layer->_layer_name = "input_pos";
                    input_pos_layer->_tensor = input_pos_tensor;
                    this->bind_device(input_pos_layer, this->_model_config._n_layer);

                    layer.insert(std::make_pair(input_pos_tensor->get_tensor_type(), input_pos_layer));
                }else {
                    input_pos_layer_iter->second->_tensor->set_shape(input_pos_tensor->get_shape());
                    input_pos_layer_iter->second->_tensor->set_buffer_data(input_pos.data(),input_pos.size() * memory::type_traits_auto[
                                              tff::core::memory::DataType::TFF_DATA_TYPE_I32]._type_size);

                }
                //
                auto input_token_embed_layer_iter = layer.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN);
                if (input_token_embed_layer_iter == layer.end()) {
                    auto input_token_layer = std::make_shared<tff::core::model::layer::ModelLayerObject>();
                    input_token_layer->_type = LLM_TENSOR_LAYER_INPUT;
                    input_token_layer->_layer_name = "input_token";
                    input_token_layer->_tensor = token_tensor;
                    this->bind_device(input_token_layer, this->_model_config._n_layer);

                    layer.insert(std::make_pair(token_tensor->get_tensor_type(), input_token_layer));
                }else {
                    input_token_embed_layer_iter->second->_tensor->set_shape(token_tensor->get_shape());
                    input_token_embed_layer_iter->second->_tensor->set_buffer_data(tokens_data.data(),
                                      tokens_data.size() * memory::type_traits_auto[
                                          tff::core::memory::DataType::TFF_DATA_TYPE_I32]._type_size);
                }
            }
        }
    }

    void LLMInferRuntime::build_output() {
        auto output_tensor = std::make_shared<tff::core::memory::Tensor>(
            tff::core::memory::DataType::TFF_DATA_TYPE_F32,
            memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
            std::array<int64_t, MAX_TENSOR_DIM>{
                static_cast<int64_t>(this->_model_config._n_embd), this->_vocabulary_ptr->get_vocab_size(), 1, 1
            }, true);

        output_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_OUTPUT);


        auto output_layer_iter = this->_layer_map.find(LLM_TENSOR_LAYER_OUTPUT);
        if (output_layer_iter != this->_layer_map.end()) {
            for (auto &layers: output_layer_iter->second) {
                auto &layer = layers.second;
                auto out_put_layer_iter = layer.find(memory::ModelTensorType::LLM_TENSOR_OUTPUT);
                if (out_put_layer_iter == layer.end()) {
                    auto out_put_layer = std::make_shared<tff::core::model::layer::ModelLayerObject>();
                    out_put_layer->_type = LLM_TENSOR_LAYER_OUTPUT;
                    out_put_layer->_layer_name = "output";
                    out_put_layer->_tensor = output_tensor;
                    this->bind_device(out_put_layer, this->_model_config._n_layer);

                    layer.insert(std::make_pair(output_tensor->get_tensor_type(), out_put_layer));
                }
            }
        }
    }

    void LLMInferRuntime::update_kv_cache(model::GraphContext &graph_ctx, std::shared_ptr<LLMBatch> &ubatch) {
        for (const auto &[layer_id, device_id] : graph_ctx._layer_device_map) {
            graph_ctx._kv_cache_ptr[device_id]->set_kv(ubatch->_token_seq_ids.back(),
                                                         layer_id,
                                                         graph_ctx._n_tokens);
        }
    }

    void LLMInferRuntime::load_hparams(bool is_fuse_op, const tff::core::memory::DataType &kv_data_type) {
        const auto &ctx = _model_loader->get_model_ctx();
        LOAD_KEY_VALUE(ctx, std::string, std::string, tff::core::model::ModelMetaKV::LLM_KV_GENERAL_ARCHITECTURE,
                       this->_arch_name);
        LOAD_KEY_VALUE(ctx, std::string, std::string, tff::core::model::ModelMetaKV::LLM_KV_GENERAL_NAME, this->_name);
        this->_model_config = this->_model_loader->get_model_config();
        this->_model_config._is_fuse_op = is_fuse_op;
        this->_model_config._kv_data_type = kv_data_type;
    }

    void LLMInferRuntime::load_vocab() const {
        this->_vocabulary_ptr->load_vocabulary(this->_model_loader);
    }

    //
    bool LLMInferRuntime::build_layers() {
        const std::string &name = this->_model_config._arch_name;
        auto &weight_map = this->_model_loader->get_weight_map();
        size_t total_layer_num = this->_model_config._n_layer;
        for (auto &weight: weight_map) {
            size_t layer_index = 0;
            auto tensor = weight.second._tensor_ptr;
            auto &layer_info = tff::core::global::LLM_LAYER_OP_INFOS.find(tensor->get_tensor_type())->second;
            if (layer_info.first == tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING) {
                auto get_layer_index = [](const std::string &layer_name) -> size_t {
                    const int pos0 = layer_name.find_first_of('.') + 1;
                    const std::string substr = layer_name.substr(pos0, layer_name.size());
                    const std::string subsubstr = substr.substr(0, substr.find_first_of("."));
                    return std::stoull(subsubstr);
                };
                layer_index = get_layer_index(weight.first);
            }
            auto get_layer_name = [](const std::string &layer_name) -> std::string {
                const int pos0 = layer_name.find_last_of('.');
                const std::string substr = layer_name.substr(0, pos0);
                return substr;
            };

            auto layer = std::make_shared<tff::core::model::layer::ModelLayerObject>();
            layer->_type = layer_info.first;
            layer->_layer_name = get_layer_name(weight.first);
            layer->_layer_index = layer_index;
            layer->_tensor = weight.second._tensor_ptr;
            layer->_tensor->set_memory_type(memory::MemoryType::TFF_MEM_TYPE_RESIDENT);
            layer->_model_file_index = weight.second._idx;
            layer->_offset = weight.second._offs;
            layer->_data_size = weight.second._alignment_size;
            this->bind_device(layer, total_layer_num);

            auto& layer_index_map = this->_layer_map[layer_info.first];
            auto& tensor_map = layer_index_map[layer_index];
            tensor_map[tensor->get_tensor_type()] = std::move(layer);
        }
        return true;
    }

    void LLMInferRuntime::bind_device(std::shared_ptr<tff::core::model::layer::ModelLayerObject> &layer_obj,
                                      const int &total_layer_index) {
        if (layer_obj->_tensor == nullptr) {
            tff::log::Logger::error("[LLMInferRuntime] layer_obj->_tensor is nullptr.");
            return;
        }
        auto get_device = [](const std::string &device_type_flag, const int &layer_index,
                             const int &total_layer_index)-> std::pair<int, std::shared_ptr<
            tff::core::device::DeviceBaseObject> > {
            auto device = tff::factory::ModuleFactory::instance()->create_shared<
                tff::core::device::DeviceBaseObject>(
                DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(device_type_flag));
            if (device == nullptr) {
                return {};
            }
            std::vector<float> device_splits;
            std::vector<int> device_list;
            device->get_device_id(device_list);
            for (size_t i = 0; i < device_list.size(); ++i) {
                size_t total_mem;
                size_t free_mem;
                device->get_device_mem(i, &free_mem, &total_mem);
                device_splits.push_back(static_cast<float>(free_mem));
            }

            //计算切分点;
            float split_sum = 0.0f;
            for (size_t i = 0; i < device_list.size(); ++i) {
                split_sum += device_splits[i];
                device_splits[i] = split_sum;
            }
            for (size_t i = 0; i < device_list.size(); ++i) {
                device_splits[i] /= split_sum;
            }
            //
            const int layer_device_id = std::upper_bound(device_splits.begin(),
                                                         device_splits.begin() + device_list.size(),
                                                         float(layer_index) / total_layer_index) -
                                        device_splits.
                                        begin();
            return {device_list[layer_device_id], device};
        };
        switch (layer_obj->_type) {
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT: {
                auto [device_id, device] = get_device(
                    DEVICE_BACKEND_TYPE_CPU, layer_obj->_layer_index, total_layer_index);
                if (device != nullptr) {
                    layer_obj->_tensor->set_allocator(device->get_device_buffer_allocator(device_id));
                    layer_obj->_device_list.insert(std::make_pair(device_id, device));
                }
                break;
            }
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT:
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING: {
                auto [device_id, device] = get_device(
                    DEVICE_BACKEND_TYPE_CUDA, layer_obj->_layer_index, total_layer_index);
                if (device != nullptr) {
                    layer_obj->_tensor->set_allocator(device->get_device_buffer_allocator(device_id));
                    layer_obj->_device_list.insert(std::make_pair(device_id, device));
                }

                break;
            }
            default:
                break;
        }
    }

    bool LLMInferRuntime::load_tensor_data() {
        if (!this->_mem_graph_ptr) {
            this->init_io_graph(this->_mem_graph_ptr);
        }

        try {
            this->_task_manager->build_task_schedule(schedule::TaskType::TFF_TASK_TYPE_IO, this->_mem_graph_ptr);
            this->_task_manager->run(tff::schedule::TaskType::TFF_TASK_TYPE_IO);
        } catch (const std::exception &e) {
            tff::log::Logger::error("Prefill forward failed: {%s}", e.what());
            return false;
        }

        return true;
    }
}
