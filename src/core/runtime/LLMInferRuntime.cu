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
        this->_vocabulary_ptr = std::make_unique<tff::core::model::LLMLLaMaVocabulary>();
        //
        this->load_hparams();
        this->load_vocab();
        this->load_layers();
        return bRet;
    }

    bool LLMInferRuntime::load_model_config(const std::string &model_config_file_path,
                                            tff::core::model::ModelConfig &params) {
        tff::core::model::ModelConfigReader::Config cfg = tff::core::model::ModelConfigReader::read(model_config_file_path);
        params._architectures = cfg.architectures;
        return true;
    }

    bool LLMInferRuntime::init_device() {
        auto gpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            DEVICE_BACKEND_TYPE_CUDA);
        if (gpu_device) {
            this->_devices.insert(gpu_device);
            this->_has_gpu_backend = true;
        }

        auto cpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            DEVICE_BACKEND_TYPE_CPU);
        if (cpu_device) {
            this->_devices.insert(cpu_device);
        }

        return this->_devices.empty() ? false : true;
    }

    bool LLMInferRuntime::init_runtime_context() {
        tff::core::memory::LLMKVCache::KVConfig kv_cfg;
        kv_cfg._n_embd_head = this->_model_config._n_embd;
        kv_cfg._n_head = this->_model_config._n_head_arr.size();
        kv_cfg._n_head_kv = this->_model_config._n_embd_head_k;
        kv_cfg._n_layer = this->_model_config._n_layer;
        kv_cfg._use_sliding_window = this->_model_config._n_swa != 0;
        const auto device = *this->_devices.begin();
        const float one_page_size = 2 * kv_cfg._n_embd_head * kv_cfg._n_head_kv * PAGE_SIZE *
                                    tff::core::memory::type_traits_auto[this->_model_config._kv_data_type]._type_size;
        std::vector<int> device_ids;
        device->get_device_id(device_ids);
        float free_mem_sum = 0;
        for (const auto device_id : device_ids) {\
            size_t free_mem = 0;
            size_t total_mem = 0;
            device->get_device_mem(device_id, &free_mem, &total_mem);
            free_mem_sum += free_mem;
        }
        free_mem_sum -= this->_model_loader->get_model_context()->_max_tensor_bytesize * 2;//计算一层
        kv_cfg._total_pages = int(free_mem_sum / one_page_size);
        this->_kv_cache_ptr = std::make_unique<tff::core::memory::LLMKVCache>(this->_model_config._kv_data_type, kv_cfg,
                                                                              device->get_device_buffer_allocator());
    }

    bool LLMInferRuntime::init_graph() {


        return true;
    }

    bool LLMInferRuntime::prefill(const std::string &prompt) {

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
        const std::string &name = std::string(tff::core::model::LLM_ARCH_NAMES.find(this->_architecture)->second);
        auto &weight_map = this->_model_loader->get_weight_map();
        size_t total_layer_num = -1;
        size_t layer_index = -1;
        for (auto &weight: weight_map) {
            std::shared_ptr<tff::core::graph::GraphNode> layer_node;
            auto callback = tff::factory::FunctionFactory::instance()->get_callback<void(
                std::shared_ptr<tff::core::memory::Tensor> &,
                std::shared_ptr<tff::core::graph::GraphNode> &,
                const size_t &, const size_t &)>(CREATE_LAYER_FLAG, name);
            auto tensor = weight.second._tensor_ptr;
            auto &layer_info = tff::core::model::LLM_LAYER_OP_INFOS.find(tensor->get_tensor_type())->second;
            if (layer_info.first == tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING) {
                total_layer_num = this->_model_config._n_layer;
                auto get_layer_index = [](const std::string &layer_name) -> size_t {
                    const int pos0 = layer_name.find_first_of(".") + 1;
                    const std::string substr = layer_name.substr(pos0, layer_name.size());
                    const std::string subsubstr = substr.substr(0, substr.find_first_of("."));
                    return std::stoull(subsubstr);
                };
                layer_index = get_layer_index(weight.first);
            }
            if (callback) {
                callback(tensor, layer_node, total_layer_num, layer_index);
                layer_node->_file_idx = weight.second._idx;
                this->_layer_map[layer_info.first].push_back(layer_node);
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
