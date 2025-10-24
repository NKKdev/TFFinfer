//
// Created by nkk on 2025/10/21.
//

#include "LLMModel.h"

namespace tff::core::model {
    bool LLMModel::load_model(const std::vector<std::string> &model_files_path,
                              const tff::core::model::ModelParams &params) const {
        bool bRet = true;
        bRet &= this->_model_loader->load(model_files_path, params._use_mmap, params._check_tensors);

        return bRet;
    }

    void LLMModel::load_hparams() {
        const auto &ctx = _model_loader->get_gguf_ctx();
#define LOAD_KEY_VALUES(DataType, key_value, dst) \
        dst = get_value<tff::core::model::ModelMetaKV,GGUFContext::BasicType, DataType>(key_value, ctx)
#define LOAD_KEY_VALUE(DataType, key_value, dst) \
        dst = LOAD_KEY_VALUES(DataType, key_value, dst)[0]

        LOAD_KEY_VALUE(std::string,tff::core::model::ModelMetaKV::LLM_KV_GENERAL_NAME,            this->_name);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_EMBEDDING_LENGTH,        this->_head_params._n_embd);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_BLOCK_COUNT,             this->_head_params._n_layer);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_EXPERT_COUNT,            this->_head_params._n_expert);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_EXPERT_USED_COUNT,       this->_head_params._n_expert_used);
        LOAD_KEY_VALUE(bool,       tff::core::model::ModelMetaKV::LLM_KV_ROPE_SCALING_FINETUNED,  this->_head_params._rope_fine_tuned);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_KEY_LENGTH,    this->_head_params._n_embd_head_k);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_VALUE_LENGTH,  this->_head_params._n_embd_head_v);
        LOAD_KEY_VALUE(uint32_t,   tff::core::model::ModelMetaKV::LLM_KV_ROPE_DIMENSION_COUNT,    this->_head_params._n_rot);
        //
        LOAD_KEY_VALUES(uint32_t,  tff::core::model::ModelMetaKV::LLM_KV_FEED_FORWARD_LENGTH,     this->_head_params._n_ff_arr);
        LOAD_KEY_VALUES(uint32_t,  tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT,    this->_head_params._n_head_arr);
        LOAD_KEY_VALUES(uint32_t,  tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT_KV, this->_head_params._n_head_kv_arr);

#undef LOAD_KEY_VALUE
#undef LOAD_KEY_VALUES
    }
}
