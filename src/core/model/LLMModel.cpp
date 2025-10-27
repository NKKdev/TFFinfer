//
// Created by nkk on 2025/10/21.
//

#include "LLMModel.h"
#include "model/base/ModelDetectorRegistry.h"
namespace tff::core::model {
    bool LLMModel::load_model(const std::vector<std::string> &model_files_path,
                              const tff::core::model::ModelConfig &params) {
        bool bRet = true;
        auto &model_detector = ModelDetectyorRegistry::get().find_dector(params._architectures);
        this->_model_loader = model_detector->create_loader();
        this->_model_loader->load_from_file(model_files_path, params._use_mmap, params._check_tensors);

        return bRet;
    }

    bool LLMModel::load_model_config(const std::string &model_config_file_path) {
        return true;
    }

    void LLMModel::load_stats() {
        //this->_n_elements = this->_model_loader->_n_elements;
        //this->_n_bytes = this->_model_loader->_n_bytes;
    }

    void LLMModel::load_hparams() {
        const auto &ctx = _model_loader->get_model_context();

        LOAD_KEY_VALUE(std::string,tff::core::model::ModelMetaKV::LLM_KV_GENERAL_NAME,            this->_name);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_EMBEDDING_LENGTH,        this->_model_config._n_embd);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_BLOCK_COUNT,             this->_model_config._n_layer);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_EXPERT_COUNT,            this->_model_config._n_expert);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_EXPERT_USED_COUNT,       this->_model_config._n_expert_used);
        LOAD_KEY_VALUE(bool,       tff::core::model::ModelMetaKV::LLM_KV_ROPE_SCALING_FINETUNED,  this->_model_config._rope_fine_tuned);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_KEY_LENGTH,    this->_model_config._n_embd_head_k);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_VALUE_LENGTH,  this->_model_config._n_embd_head_v);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_ROPE_DIMENSION_COUNT,    this->_model_config._n_rot);
        //
        LOAD_KEY_VALUES(uint32_t,  tff::core::model::ModelMetaKV::LLM_KV_FEED_FORWARD_LENGTH,     this->_model_config._n_ff_arr);
        LOAD_KEY_VALUES(uint32_t,  tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT,    this->_model_config._n_head_arr);
        LOAD_KEY_VALUES(uint32_t,  tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT_KV, this->_model_config._n_head_kv_arr);

    }

    void LLMModel::load_vocab() const {
        this->_vocabulary_ptr->load_vocabulary(_model_loader);
    }
    //
    bool LLMModel::load_layers() {

        return true;
    }
}
