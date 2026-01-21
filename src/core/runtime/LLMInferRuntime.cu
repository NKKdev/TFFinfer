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
        if (model_files_path.empty()) {
            return false;
        }
        bool bRet = true;
        auto model_detector = tff::core::model::ModelDetectorRegistry::find_dector(
            tff::utils::get_file_ext(model_files_path[0]));

        this->_model_loader = model_detector->create_loader();
        this->_model_loader->load_from_file(model_files_path, params);
        this->build_model_creator();
        this->_vocabulary_ptr = std::make_unique<tff::core::model::LLMLLaMaVocabulary>();

        this->load_hparams(params._is_fuse_op, params._kv_data_type);
        this->load_vocab();
        this->build_layers();
        this->load_tensor_data();
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
        this->_model_creator->_model_ctx._n_embd_head = cfg._n_embd / cfg._n_head_arr[0];
        this->_model_creator->_model_ctx._max_seq_len = cfg._n_ctx;
        this->_model_creator->_model_ctx._n_embd_head_k = cfg._n_embd_head_k;
        this->_model_creator->_model_ctx._n_embd_head_v = cfg._n_embd_head_v;
        this->_model_creator->_model_ctx._n_head = cfg._n_head_arr[0];
        this->_model_creator->_model_ctx._n_head_kv = cfg._n_head_kv_arr[0];
        this->_model_creator->_model_ctx._n_layer = cfg._n_layer;
        this->_model_creator->_model_ctx._use_fp16 = cfg._use_f16;
        this->_model_creator->_model_ctx._use_mmap = cfg._use_mmap;
        this->_model_creator->_model_ctx._rope_freq_base = cfg._rope_freq_base;
        this->_model_creator->_model_ctx._rope_freq_scale = cfg._rope_freq_scale;
        this->_model_creator->_model_ctx._model_loader = this->_model_loader;
        this->_model_creator->_model_ctx._f_norm_rms_eps = cfg._f_norm_rms_eps;
        return true;
    }

    bool LLMInferRuntime::init_device() {
        auto gpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
        if (gpu_device) {
            gpu_device->device_init();
            this->_devices.insert(gpu_device);
            this->_has_gpu_backend = true;
            std::vector<int> device_ids;
            gpu_device->get_device_id(device_ids);
            for (auto device_id: device_ids) {
                this->_devices_map.insert(std::make_pair(device_id, gpu_device));
            }
        }

        auto cpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
        if (cpu_device) {
            cpu_device->device_init();
            this->_devices.insert(cpu_device);
            std::vector<int> device_ids;
            cpu_device->get_device_id(device_ids);
            for (auto device_id: device_ids) {
                this->_devices_map.insert(std::make_pair(device_id, cpu_device));
            }
        }

        return !(this->_devices.empty());
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
        kv_cfg._data_type = this->_model_config._kv_data_type;
        const auto device = *this->_devices.begin(); //优先构建设备优先级高的kv cache
        if (!device) {
            tff::log::Logger::error("No valid device found in _devices.");
            return false;
        }
        //
        auto type_size = memory::type_traits_auto[kv_cfg._data_type]._type_size;

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

        for (const auto device_id: device_ids) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            device->get_device_mem(device_id, &free_mem, &total_mem);
            tff::log::Logger::info("Device {%d}: Total memory: {%lld} bytes, Free memory: {%lld} bytes",
                                   device_id, total_mem, free_mem);
            //

            //预留模型上下文至少MAX_PREFETCH_BUFFER_SIZE层权重和其他开销的显存;
            const size_t context_reserve = this->_model_loader->get_model_ctx()->_max_tensor_byte_size *
                                           MAX_PREFETCH_BUFFER_SIZE;
            free_mem -= context_reserve;
            tff::log::Logger::info("Reserved memory for model context and overhead: {%lld} bytes", context_reserve);

            if (free_mem <= 0) {
                tff::log::Logger::error(
                    "Insufficient GPU memory. After reservation, free memory is {%lf} bytes (<= 0).",
                    free_mem);
                return false;
            }

            //计算总页数并创建KV Cache
            kv_cfg._total_pages = min(static_cast<int>(free_mem / one_page_size), kv_cfg._max_tokens / PAGE_SIZE);
            tff::log::Logger::info("KV Cache: Total available free memory for KV: {%lld} bytes",
                                   static_cast<size_t>(free_mem));
            tff::log::Logger::info("KV Cache: Total pages calculated: {%d} ({%lld} bytes per page)",
                                   kv_cfg._total_pages, static_cast<size_t>(one_page_size));

            if (kv_cfg._total_pages == 0) {
                tff::log::Logger::error(
                    "Calculated total KV cache pages is 0. Available memory ({%lld}) is less than one page size ({%lld}).",
                    static_cast<size_t>(free_mem), static_cast<size_t>(one_page_size));
                return false;
            }

            try {
                this->_kv_cache_ptr[device_id] = std::make_shared<tff::core::memory::LLMKVCache>(
                    this->_model_config._kv_data_type,
                    kv_cfg,
                    device->get_device_buffer_allocator(device_id)
                );
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

        return true; // 初始化成功
    }

    bool LLMInferRuntime::init_graph() {
        if (this->_layer_map.empty()) {
            tff::log::Logger::error("model layer is invalid!!\n");
            return false;
        }
        tff::log::Logger::info("Initializing graph");

        this->_model_creator->build_graph(this->_layer_map, this->_infer_graph_ptr);
        return true;
    }

    bool LLMInferRuntime::init_io_graph() {
        if (this->_layer_map.empty()) {
            tff::log::Logger::error("model layer is invalid!!\n");
            return false;
        }
        tff::log::Logger::info("Initializing io graph");

        this->_model_creator->build_mem_graph(this->_layer_map, this->_mem_graph_ptr);
        return true;
    }

    void LLMInferRuntime::build_mem_offset(const std::shared_ptr<tff::core::runtime::LLMMemManager> &_mem_manager_ptr,
                                           const std::shared_ptr<graph::Graph> &graph_ptr,
                                           std::unordered_map<std::string, std::unordered_map<int, size_t> > &
                                           mem_buffer_offset_map,
                                           const MemoryType &type) const {
        for (auto &node: graph_ptr->total_nodes()) {
            auto current_device = *node->device().begin();
            auto [start, end] = graph_ptr->get_lifetime(node);
            tff::log::Logger::info("node： %s start: %d, end: %d", node->name().c_str(), start, end);
            auto mem_offset = _mem_manager_ptr->allocate_memory(node->get_tensor()->get_bytes(),
                                                                start, end, current_device.first, type);
            node->get_tensor()->set_external_memory_index(mem_offset);
            auto device_type_flag = current_device.second->get_device_type_flag(current_device.first);
            auto iter = mem_buffer_offset_map.find(device_type_flag);
            if (iter != mem_buffer_offset_map.end()) {
                auto devoce_iter = iter->second.find(current_device.first);
                if (devoce_iter != iter->second.end()) {
                    devoce_iter->second = devoce_iter->second < mem_offset ? mem_offset : devoce_iter->second;
                } else {
                    mem_buffer_offset_map[device_type_flag].insert(std::make_pair(current_device.first, mem_offset));
                }
            } else {
                std::unordered_map<int, size_t> device_mem_offset_map;
                device_mem_offset_map.insert(std::make_pair(current_device.first, mem_offset));
                mem_buffer_offset_map.insert(std::make_pair(device_type_flag, device_mem_offset_map));
            }
        }
    }

    bool LLMInferRuntime::init_mem_manager(const std::shared_ptr<graph::Graph> &graph_ptr,
                                           std::shared_ptr<tff::core::runtime::LLMMemManager> &_mem_manager_ptr,
                                           const MemoryType &type) const {
        if (graph_ptr == nullptr) {
            tff::log::Logger::error("model graph is invalid!!\n");
            return false;
        }
        if (_mem_manager_ptr == nullptr) {
            tff::log::Logger::error("model memory manager is invalid!!\n");
            return false;
        }
        std::unordered_map<std::string, std::unordered_map<int, size_t> > mem_buffer_offset_map;
        this->build_mem_offset(_mem_manager_ptr, graph_ptr, mem_buffer_offset_map, type);

        bool bRet = true;
        for (auto &mem_buffer_offset: mem_buffer_offset_map) {
            for (auto &mem_buffer: mem_buffer_offset.second) {
                bRet &= _mem_manager_ptr->init(mem_buffer.first, type);
            }
        }

        this->_model_creator->_model_ctx._mem_manager_ptr = _mem_manager_ptr;
        return bRet;
    }

    bool LLMInferRuntime::prefill() {
        for (auto &batch: this->_llm_batch_manager_ptr->_ubatches) {
            this->build_inputs(batch);
            this->build_output();
            if (!this->_infer_graph_ptr) {
                this->init_graph();
                //init mem buffer;
                if (!this->init_mem_manager(this->_infer_graph_ptr, this->_mem_manager_ptr, MemoryType::ACTIVATION)) {
                    tff::log::Logger::error("Failed to initialize memory manager.");
                    return false;
                }
            }
            try {
                this->_task_manager->build_task_schedule(tff::schedule::TaskType::TFF_TASK_TYPE_INFER,
                                                         this->_infer_graph_ptr, this->_model_config._is_fuse_op);
                this->_task_manager->run(tff::schedule::TaskType::TFF_TASK_TYPE_INFER);
            } catch (const std::exception &e) {
                tff::log::Logger::error("Prefill forward failed: {%s}", e.what());
                return false;
            }
        }
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
        auto token_tensor = std::make_shared<tff::core::memory::Tensor>(
            MAX_TENSOR_DIM, tff::core::memory::DataType::TFF_DATA_TYPE_I32,
            std::array<int64_t, MAX_TENSOR_DIM>{
                static_cast<int64_t>(tokens_data.size()), 1, 1, 1
            }, true);
        token_tensor->set_buffer_data(tokens_data.data(),
                                      tokens_data.size() * memory::type_traits_auto[
                                          tff::core::memory::DataType::TFF_DATA_TYPE_I32]._type_size);
        token_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN);

        auto input_pos_tensor = std::make_shared<tff::core::memory::Tensor>(MAX_TENSOR_DIM,
                                                                            tff::core::memory::DataType::TFF_DATA_TYPE_I32,
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
                }
            }
        }
    }

    void LLMInferRuntime::build_output() {
        auto output_tensor = std::make_shared<tff::core::memory::Tensor>(
            MAX_TENSOR_DIM, tff::core::memory::DataType::TFF_DATA_TYPE_F32,
            std::array<int64_t, MAX_TENSOR_DIM>{
                static_cast<int64_t>(this->_model_config._n_embd), this->_vocabulary_ptr->get_vocab_size(), 1, 1
            }, true);

        output_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_OUTPUT);


        auto input_layer_iter = this->_layer_map.find(LLM_TENSOR_LAYER_OUTPUT);
        if (input_layer_iter != this->_layer_map.end()) {
            for (auto &layers: input_layer_iter->second) {
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

    void LLMInferRuntime::load_stats() {
        //this->_n_elements = this->_model_loader->_n_elements;
        //this->_n_bytes = this->_model_loader->_n_bytes;
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
            std::shared_ptr<tff::core::graph::GraphNode> layer_node;
            auto tensor = weight.second._tensor_ptr;
            if (tensor->get_tensor_type() == memory::ModelTensorType::LLM_TENSOR_ATTN_K_NORM) {
                int tmp = 9;
            }
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
            layer->_tensor = weight.second._tensor_ptr;
            layer->_model_file_index = weight.second._idx;
            layer->_offset = weight.second._offs;
            layer->_data_size = weight.second._alignment_size;
            this->bind_device(layer, total_layer_num);

            auto iter = this->_layer_map[layer_info.first].find(layer_index);
            if (iter != this->_layer_map[layer_info.first].end()) {
                iter->second.insert(std::make_pair(tensor->get_tensor_type(), layer));
            } else {
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                            tff::core::model::layer::ModelLayerObject> >
                        tensor_type_graph_map;
                tensor_type_graph_map.insert(std::make_pair(tensor->get_tensor_type(), layer));
                this->_layer_map[layer_info.first].insert(std::make_pair(layer_index, tensor_type_graph_map));
            }
        }
        return true;
    }

    void LLMInferRuntime::bind_device(std::shared_ptr<tff::core::model::layer::ModelLayerObject> &layer_obj,
                                      const int &total_layer_index) {
        auto get_device = [](const std::string &device_type_flag, const int &layer_index,
                             const int &total_layer_index)-> std::unordered_map<int, std::shared_ptr<
            tff::core::device::DeviceBaseObject> > {
            auto device = tff::factory::ModuleFactory::instance()->create_shared<
                tff::core::device::DeviceBaseObject>(
                DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(device_type_flag));
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
            return {{device_list[layer_device_id], device}};
        };
        switch (layer_obj->_type) {
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT: {
                layer_obj->_device_list = get_device(
                    DEVICE_BACKEND_TYPE_CPU, layer_obj->_layer_index, total_layer_index);
                break;
            }
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT:
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING: {
                layer_obj->_device_list = get_device(
                    DEVICE_BACKEND_TYPE_CUDA, layer_obj->_layer_index, total_layer_index);
                break;
            }
            default:
                break;
        }
    }

    bool LLMInferRuntime::load_tensor_data() {
        this->init_io_graph();

        if (!this->init_mem_manager(this->_mem_graph_ptr, this->_mem_manager_ptr, MemoryType::WEIGHT)) {
            tff::log::Logger::error("Failed to initialize memory manager.");
            return false;
        }

        try {
            this->_task_manager->build_task_schedule(tff::schedule::TaskType::TFF_TASK_TYPE_IO, this->_mem_graph_ptr);
            this->_task_manager->run(tff::schedule::TaskType::TFF_TASK_TYPE_IO);
        } catch (const std::exception &e) {
            tff::log::Logger::error("Prefill forward failed: {%s}", e.what());
            return false;
        }

        return true;
    }
}
