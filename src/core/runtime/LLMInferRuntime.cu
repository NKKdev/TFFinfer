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
        bool bRet = true;
        auto model_detector = tff::core::model::ModelDetectyorRegistry::get().find_dector(params._architectures);
        this->_architecture = model_detector->arch();
        this->_model_loader = model_detector->create_loader();
        this->_model_loader->load_from_file(model_files_path, params._use_mmap, params._check_tensors);
        this->_model_creator = model_detector->create_creator();
        this->_model_creator->set_loader(this->_model_loader);
        this->_vocabulary_ptr = std::make_unique<tff::core::model::LLMLLaMaVocabulary>();
        //
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
        kv_cfg._n_embd_head = this->_model_config._n_embd;
        kv_cfg._n_head = this->_model_config._n_head_arr.size();
        kv_cfg._n_head_kv = this->_model_config._n_embd_head_k;
        kv_cfg._n_layer = this->_model_config._n_layer;
        kv_cfg._use_sliding_window = this->_model_config._n_swa != 0;

        const auto device = *this->_devices.begin();
        if (!device) {
            tff::log::Logger::error("No valid device found in _devices.");
            return false;
        }
        //
        const float one_page_size = 2 * kv_cfg._n_embd_head * kv_cfg._n_head_kv * PAGE_SIZE *
                                    tff::core::memory::type_traits_auto[this->_model_config._kv_data_type]._type_size;
        tff::log::Logger::info("KV Cache: Size per page: {:.2f} bytes", one_page_size);

        //
        std::vector<int> device_ids;
        device->get_device_id(device_ids);
        if (device_ids.empty()) {
            tff::log::Logger::error("Failed to get device IDs.");
            return false;
        }
        tff::log::Logger::info("KV Cache: Target devices: {}", device_ids);

        float free_mem_sum = 0;
        for (const auto device_id: device_ids) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            device->get_device_mem(device_id, &free_mem, &total_mem);
            tff::log::Logger::info("Device {}: Total memory: {} bytes, Free memory: {} bytes",
                                   device_id, total_mem, free_mem);
            free_mem_sum += free_mem;
        }

        //预留模型上下文至少一层权重和其他开销的显存;
        const size_t context_reserve = this->_model_loader->get_model_context()->_max_tensor_bytesize * 2;
        free_mem_sum -= context_reserve;
        tff::log::Logger::info("Reserved memory for model context and overhead: {} bytes", context_reserve);

        if (free_mem_sum <= 0) {
            tff::log::Logger::error("Insufficient GPU memory. After reservation, free memory is {} bytes (<= 0).",
                                    free_mem_sum);
            return false;
        }

        //计算总页数并创建KV Cache
        kv_cfg._total_pages = static_cast<int>(free_mem_sum / one_page_size);
        tff::log::Logger::info("KV Cache: Total available free memory for KV: {} bytes",
                               static_cast<size_t>(free_mem_sum));
        tff::log::Logger::info("KV Cache: Total pages calculated: {} ({} bytes per page)",
                               kv_cfg._total_pages, static_cast<size_t>(one_page_size));

        if (kv_cfg._total_pages == 0) {
            tff::log::Logger::error(
                "Calculated total KV cache pages is 0. Available memory ({}) is less than one page size ({}).",
                static_cast<size_t>(free_mem_sum), static_cast<size_t>(one_page_size));
            return false;
        }

        try {
            this->_kv_cache_ptr = std::make_unique<tff::core::memory::LLMKVCache>(
                this->_model_config._kv_data_type,
                kv_cfg,
                device->get_device_buffer_allocator()
            );
            tff::log::Logger::info("KV Cache successfully initialized with {} pages.", kv_cfg._total_pages);
        } catch (const std::exception &e) {
            tff::log::Logger::error("Failed to create LLMKVCache instance. Exception: {}", e.what());
            return false;
        } catch (...) {
            tff::log::Logger::error("Failed to create LLMKVCache instance. Unknown exception occurred.");
            return false;
        }
        //
        this->init_graph();
        tff::log::Logger::info("LLMInferRuntime context initialized successfully.");
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

    bool LLMInferRuntime::prefill(const std::vector<std::string> &prompt_batches) {
        if (prompt_batches.empty()) {
            tff::log::Logger::error("Prompt is empty.");
            return false;
        }

        const size_t batch_size = prompt_batches.size();
        std::vector<std::vector<int> > tokenized_batch;
        size_t max_seq_len = 0;
        for (size_t i = 0; i < batch_size; ++i) {
            std::vector<int> tokens;
            this->_vocabulary_ptr->tokenize(prompt_batches[i], tokens);

            if (tokens.size() > this->_model_config._n_ctx) {
                tff::log::Logger::warning("Prompt {} length ({}) exceeds context length ({}). Truncating.",
                                          i, tokens.size(), this->_model_config._n_ctx);
                tokens.resize(this->_model_config._n_ctx);
            }
            max_seq_len = std::max(max_seq_len, tokens.size());
            tokenized_batch.push_back(std::move(tokens));
            tff::log::Logger::info("Batch {}: tokenized {} tokens.", i, tokens.size());
        }

        for (auto &tokens: tokenized_batch) {
            tokens.resize(max_seq_len, 0);
        }

        std::vector<int32_t> flat_tokens;
        flat_tokens.reserve(batch_size * max_seq_len);
        for (const auto &tokens: tokenized_batch) {
            flat_tokens.insert(flat_tokens.end(), tokens.begin(), tokens.end());
        }

        auto input_tensor = std::make_shared<tff::core::memory::Tensor>(
            tff::core::memory::DataType::TFF_DATA_TYPE_I32,
            std::vector<int64_t>{static_cast<int64_t>(batch_size), static_cast<int64_t>(max_seq_len)}
        );

        const size_t total_bytes = flat_tokens.size() *
                                   tff::core::memory::type_traits_auto[tff::core::memory::DataType::TFF_DATA_TYPE_I32].
                                   _type_size;
        input_tensor->set_buffer_data(flat_tokens.data(), total_bytes);

        auto input_node = this->_graph_ptr->get_input_nodes();
        input_node->set_inputs({input_tensor});

        this->_kv_cache_ptr->begine_prefill(batch_size, max_seq_len);

        try {
            this->_graph_ptr->forward();
        } catch (const std::exception &e) {
            tff::log::Logger::error("Prefill forward failed: {}", e.what());
            return false;
        }

        // Step 7: finalize KV cache
        this->_kv_cache_ptr->end_prefill();

        tff::log::Logger::info("Prefill completed successfully for {} batches, max seq len = {}.",

                               batch_size, max_seq_len);

        return true;
    }

    bool LLMInferRuntime::decode(const int &n_predict, std::string &generate_str) {
        return true;
    }

    void LLMInferRuntime::load_stats() {
        //this->_n_elements = this->_model_loader->_n_elements;
        //this->_n_bytes = this->_model_loader->_n_bytes;
    }

    void LLMInferRuntime::load_hparams() {
        const auto &ctx = _model_loader->get_model_context();
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
                    const int pos0 = layer_name.find_first_of(".") + 1;
                    const std::string substr = layer_name.substr(pos0, layer_name.size());
                    const std::string subsubstr = substr.substr(0, substr.find_first_of("."));
                    return std::stoull(subsubstr);
                };
                layer_index = get_layer_index(weight.first);
            }
            if (this->_model_creator) {
                this->_model_creator->build_layer(tensor, layer_node, total_layer_num, layer_index);
                if (!layer_node) {
                    tff::log::Logger::error("current layer %s create failed!! \n", weight.first.c_str());
                    continue;
                }
                layer_node->set_file_idx(weight.second._idx);
                layer_node->set_name(weight.first);
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
