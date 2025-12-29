//
// Created by nkk on 2025/10/21.
//

#include "LLMInferRuntime.h"
#include "model/base/ModelDetectorRegistry.h"
#include "model/llama/LLAMACreator.h"
#include "FunctionFactory.h"
#include "model/base/ModelConfigReader.h"
#include "mem/BaseDefine.h"

namespace tff::core::runtime {
    bool LLMInferRuntime::load_model(const std::vector<std::string> &model_files_path,
                                     const tff::core::model::ModelConfig &params) {
        this->_model_config._is_fuse_op = params._is_fuse_op;
        bool bRet = true;
        auto model_detector = tff::core::model::ModelDetectyorRegistry::get().find_dector(params._architectures);
        this->_architecture = model_detector->arch();
        this->_model_loader = model_detector->create_loader();
        this->_model_loader->load_from_file(model_files_path, params._use_mmap, params._check_tensors);
        this->_model_creator = model_detector->create_creator();
        this->_model_creator->set_loader(this->_model_loader);
        this->_vocabulary_ptr = std::make_unique<tff::core::model::LLMLLaMaVocabulary>();

        this->load_hparams();
        this->load_vocab();
        this->load_layers();
        return bRet;
    }

    bool LLMInferRuntime::load_model_config(const std::string &model_config_file_path,
                                            tff::core::model::ModelConfig &params) {
        tff::core::model::ModelConfigReader::Config cfg = tff::core::model::ModelConfigReader::read(
            model_config_file_path);
        params._architectures = cfg.architectures;
        return true;
    }

    bool LLMInferRuntime::init_device() {
        auto gpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
        if (gpu_device) {
            this->_devices.insert(gpu_device);
            this->_has_gpu_backend = true;
        }

        auto cpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
        if (cpu_device) {
            this->_devices.insert(cpu_device);
        }

        return this->_devices.empty() ? false : true;
    }

    bool LLMInferRuntime::init_runtime_context() {
        //
        tff::core::memory::LLMKVCache::KVConfig kv_cfg;
        kv_cfg._n_embd_head = this->_model_config._n_embd_head_k;
        kv_cfg._n_head = this->_model_config._n_head_arr[0];
        kv_cfg._n_head_kv = this->_model_config._n_head_kv_arr[0];
        kv_cfg._n_layer = this->_model_config._n_layer;
        kv_cfg._use_sliding_window = this->_model_config._n_swa != 0;
        kv_cfg._max_tokens = this->_model_config._n_ctx;
        kv_cfg._use_f16 = this->_model_config._use_f16;
        const auto device = *this->_devices.begin();
        if (!device) {
            tff::log::Logger::error("No valid device found in _devices.");
            return false;
        }
        //
        auto type_size = kv_cfg._use_f16 ? memory::type_traits_auto[memory::DataType::TFF_DATA_TYPE_F16]._type_size:
        memory::type_traits_auto[memory::DataType::TFF_DATA_TYPE_F32]._type_size;//todo f32 unimplement

        const float one_page_size = kv_cfg._n_embd_head * kv_cfg._n_head_kv * PAGE_SIZE * type_size;
        tff::log::Logger::info("KV Cache: Size per page: {%lf} bytes", one_page_size);

        //
        std::vector<int> device_ids;
        device->get_device_id(device_ids);
        if (device_ids.empty()) {
            tff::log::Logger::error("Failed to get device IDs.");
            return false;
        }
        tff::log::Logger::info("KV Cache: Target devices: {%d}", device_ids[0]);

        float free_mem_sum = 0;
        for (const auto device_id: device_ids) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            device->get_device_mem(device_id, &free_mem, &total_mem);
            tff::log::Logger::info("Device {%d}: Total memory: {%lld} bytes, Free memory: {%lld} bytes",
                                   device_id, total_mem, free_mem);
            free_mem_sum += free_mem;
        }

        //预留模型上下文至少MAX_PREFETCH_BUFFER_SIZE层权重和其他开销的显存;
        const size_t context_reserve = this->_model_loader->get_model_ctx()->_max_tensor_byte_size *
                                       MAX_PREFETCH_BUFFER_SIZE;
        free_mem_sum -= context_reserve;
        tff::log::Logger::info("Reserved memory for model context and overhead: {%lld} bytes", context_reserve);

        if (free_mem_sum <= 0) {
            tff::log::Logger::error("Insufficient GPU memory. After reservation, free memory is {%lf} bytes (<= 0).",
                                    free_mem_sum);
            return false;
        }

        //计算总页数并创建KV Cache
        kv_cfg._total_pages = min(static_cast<int>(free_mem_sum / one_page_size), kv_cfg._max_tokens / PAGE_SIZE);
        tff::log::Logger::info("KV Cache: Total available free memory for KV: {%lld} bytes",
                               static_cast<size_t>(free_mem_sum));
        tff::log::Logger::info("KV Cache: Total pages calculated: {%d} ({%lld} bytes per page)",
                               kv_cfg._total_pages, static_cast<size_t>(one_page_size));

        if (kv_cfg._total_pages == 0) {
            tff::log::Logger::error(
                "Calculated total KV cache pages is 0. Available memory ({%lld}) is less than one page size ({%lld}).",
                static_cast<size_t>(free_mem_sum), static_cast<size_t>(one_page_size));
            return false;
        }

        try {
            this->_kv_cache_ptr = std::make_shared<tff::core::memory::LLMKVCache>(
                this->_model_config._kv_data_type,
                kv_cfg,
                device->get_device_buffer_allocator()
            );
            tff::log::Logger::info("KV Cache successfully initialized with {%d} pages.", kv_cfg._total_pages);
        } catch (const std::exception &e) {
            tff::log::Logger::error("Failed to create LLMKVCache instance. Exception: {%s}", e.what());
            return false;
        } catch (...) {
            tff::log::Logger::error("Failed to create LLMKVCache instance. Unknown exception occurred.");
            return false;
        }
        if (this->_weight_mem_manager_ptr == nullptr) {
            tff::log::Logger::info("Memory Manager created failed.");
            return false;
        }

        //init weight mem buffer;
        if (this->_weight_mem_manager_ptr->init(this->_model_loader->get_model_ctx()->_max_tensor_byte_size)) {
            tff::log::Logger::info("LLMInferRuntime context initialized successfully.");
            return true;
        } else {
            tff::log::Logger::info("LLMInferRuntime context initialized failed.");
            return false;
        }

        return true; // 初始化成功
    }

    bool LLMInferRuntime::init_graph() {
        if (this->_layer_map.empty()) {
            tff::log::Logger::error("model layer is invalid!!\n");
            return false;
        }
        tff::log::Logger::info("Initializing graph");

        this->_model_creator->build_graph(this->_layer_map, this->_graph_ptr);

        return true;
    }

    bool LLMInferRuntime::prefill() {
        this->_kv_cache_ptr->begine_prefill(this->_llm_batch_manager_ptr->_ubatches.size(), this->_llm_batch_manager_ptr->_max_seq_size);
        for (auto &batch: this->_llm_batch_manager_ptr->_ubatches) {
            this->build_inputs(batch);
            if (!this->_graph_ptr) {
                this->init_graph();
            }
            try {
                this->_task_manager->build_task_schedule(this->_model_config._is_fuse_op, this->_graph_ptr);
                this->_task_manager->run();
            } catch (const std::exception &e) {
                tff::log::Logger::error("Prefill forward failed: {%s}", e.what());
                return false;
            }
        }
        this->_kv_cache_ptr->end_prefill();
        return true;
    }

    bool LLMInferRuntime::decode(const int &n_predict, std::string &generate_str) {
        return true;
    }

    int LLMInferRuntime::encode(const std::vector<std::string> &prompt_batches) {
        if (prompt_batches.empty()) {
            tff::log::Logger::error("Prompt is empty.");
            return -1;
        }

        const size_t batch_size = prompt_batches.size();
        std::unordered_map<int, std::string> seq_prompts;
        //std::vector<std::vector<int> > tokenized_batch;
        size_t max_seq_len = 0;
        for (size_t i = 0; i < batch_size; ++i) {
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
        auto &tokens_data = batch->_tokens;
        auto &input_pos = batch->_pos;
        auto token_tensor = std::make_shared<tff::core::memory::Tensor>(MAX_TENSOR_DIM,tff::core::memory::DataType::TFF_DATA_TYPE_I32,
                                                                        std::array<int64_t, MAX_TENSOR_DIM>{
                                                                            static_cast<int64_t>(tokens_data.size())
                                                                        ,1,1,1}, true);
        token_tensor->set_buffer_data(tokens_data.data(),
                                      tokens_data.size() * memory::type_traits_auto[
                                          tff::core::memory::DataType::TFF_DATA_TYPE_I32]._type_size);
        token_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN);

        auto input_pos_tensor = std::make_shared<tff::core::memory::Tensor>(MAX_TENSOR_DIM,
            tff::core::memory::DataType::TFF_DATA_TYPE_I32,
            std::array<int64_t, MAX_TENSOR_DIM>{static_cast<int64_t>(input_pos.size()),1,1,1}, true);
        input_pos_tensor->set_buffer_data(input_pos.data(),
                                          input_pos.size() * memory::type_traits_auto[
                                              tff::core::memory::DataType::TFF_DATA_TYPE_I32]._type_size);
        input_pos_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_TOKEN_POS);

        auto input_layer_iter = this->_layer_map.find(LLM_TENSOR_LAYER_INPUT);
        if (input_layer_iter != this->_layer_map.end()) {
            for (auto &layers: input_layer_iter->second) {
                auto &layer = layers.second;
                auto input_pos_layer = layer.find(memory::ModelTensorType::LLM_TENSOR_TOKEN_POS);
                if (input_pos_layer == layer.end()) {
                    std::shared_ptr<tff::core::graph::GraphNode> layer_node;
                    this->_model_creator->build_layer(input_pos_tensor, layer_node);
                    layer.insert(std::make_pair(input_pos_tensor->get_tensor_type(), layer_node));
                }
                //
                auto input_token_embed_layer = layer.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN);
                if (input_token_embed_layer == layer.end()) {
                    std::shared_ptr<tff::core::graph::GraphNode> layer_node;
                    this->_model_creator->build_layer(token_tensor, layer_node);
                    layer.insert(std::make_pair(token_tensor->get_tensor_type(), layer_node));
                }
            }
        }
    }

    void LLMInferRuntime::load_stats() {
        //this->_n_elements = this->_model_loader->_n_elements;
        //this->_n_bytes = this->_model_loader->_n_bytes;
    }

    void LLMInferRuntime::load_hparams() {
        const auto &ctx = _model_loader->get_model_ctx();
        LOAD_KEY_VALUE(std::string, std::string, tff::core::model::ModelMetaKV::LLM_KV_GENERAL_ARCHITECTURE,
                       this->_arch_name);
        LOAD_KEY_VALUE(std::string, std::string, tff::core::model::ModelMetaKV::LLM_KV_GENERAL_NAME, this->_name);
        LOAD_KEY_VALUE(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_EMBEDDING_LENGTH,
                       this->_model_config._n_embd);
        LOAD_KEY_VALUE(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_BLOCK_COUNT,
                       this->_model_config._n_layer);
        LOAD_KEY_VALUE(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_EXPERT_COUNT,
                       this->_model_config._n_expert);
        LOAD_KEY_VALUE(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_EXPERT_USED_COUNT,
                       this->_model_config._n_expert_used);
        LOAD_KEY_VALUE(bool, bool, tff::core::model::ModelMetaKV::LLM_KV_ROPE_SCALING_FINETUNED,
                       this->_model_config._rope_fine_tuned);
        LOAD_KEY_VALUE(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_KEY_LENGTH,
                       this->_model_config._n_embd_head_k);
        LOAD_KEY_VALUE(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_VALUE_LENGTH,
                       this->_model_config._n_embd_head_v);
        LOAD_KEY_VALUE(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_ROPE_DIMENSION_COUNT,
                       this->_model_config._n_rot);
        LOAD_KEY_VALUES(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_FEED_FORWARD_LENGTH,
                        this->_model_config._n_ff_arr);
        LOAD_KEY_VALUES(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT,
                        this->_model_config._n_head_arr);
        LOAD_KEY_VALUES(uint32_t, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT_KV,
                        this->_model_config._n_head_kv_arr);
    }

    void LLMInferRuntime::load_vocab() const {
        this->_vocabulary_ptr->load_vocabulary(this->_model_loader);
    }

    //
    bool LLMInferRuntime::load_layers() {
        const std::string &name = std::string(tff::core::global::LLM_ARCH_NAMES.find(this->_architecture)->second);
        auto &weight_map = this->_model_loader->get_weight_map();
        size_t total_layer_num = this->_model_config._n_layer;

        for (auto &weight: weight_map) {
            size_t layer_index = 0;
            std::shared_ptr<tff::core::graph::GraphNode> layer_node;
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
            if (this->_model_creator) {
                this->_model_creator->build_layer(tensor, layer_node, total_layer_num, layer_index);
                if (!layer_node) {
                    tff::log::Logger::error("current layer %s create failed!! \n", weight.first.c_str());
                    continue;
                }
                layer_node->set_file_idx(weight.second._idx);
                layer_node->set_name(get_layer_name(weight.first));
                layer_node->get_params()->set_param(0, get_layer_name(weight.first));
                layer_node->get_params()->set_param(1, weight.second._idx);
                layer_node->get_params()->set_param(2, weight.second._offs);
                layer_node->get_params()->set_param(3, weight.second._byte_size);
                layer_node->get_params()->set_param(4, this->_model_loader);
                auto iter = this->_layer_map[layer_info.first].find(layer_index);
                if (iter != this->_layer_map[layer_info.first].end()) {
                    iter->second.insert(std::make_pair(tensor->get_tensor_type(), layer_node));
                } else {
                    std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                tff::core::graph::GraphNode> >
                            tensor_type_graph_map;
                    tensor_type_graph_map.insert(std::make_pair(tensor->get_tensor_type(), layer_node));
                    this->_layer_map[layer_info.first].insert(std::make_pair(layer_index, tensor_type_graph_map));
                }

                const auto iter_tmp = this->_layer_map.find(
                    tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
                if (iter_tmp != this->_layer_map.end()) {
                    auto iter_tensor = iter_tmp->second.begin()->second.find(
                        tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K);
                    if (iter_tensor != iter_tmp->second.begin()->second.end()) {
                        this->_model_config._kv_data_type = iter_tensor->second.get()->data_type();
                    }
                }
            }
        }
        return true;
    }

    bool LLMInferRuntime::load_tensor_data() {
        if (this->_model_config._use_mmap) {
        } else {
        }
    }
}
