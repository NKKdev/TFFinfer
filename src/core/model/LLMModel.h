//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_LLMMODEL_H
#define TFFINFER_LLMMODEL_H
#include <unordered_map>
#include <memory>
#include "BaseDefine.h"
#include "LLMVocabulary.h"
#include "mem/Tensor.h"
#include "device/DeviceBaseObject.h"
#include "LLMModelLoader.h"
#include "ModuleFactory.h"
#include "ModelGlobalVar.h"

namespace tff::core::model {
    class LLMModel {
    public:
        LLMModel() {
            _model_loader = std::dynamic_pointer_cast<tff::core::model::LLMModelLoader>(
                tff::factory::ModuleFactory::instance()->create_shared("MODEL", "LLAMA"));
        }

        ~LLMModel();

    public:
        //
        bool load_model(const std::vector<std::string> &model_files_path,
                        const tff::core::model::ModelParams &params) const;

    protected:
        void load_stats();

        inline void load_arch() {
            _arch_name = get_value<tff::core::model::ModelMetaKV,
            GGUFContext::BasicType, std::string>(tff::core::model::ModelMetaKV::LLM_KV_GENERAL_ARCHITECTURE,
                _model_loader->get_gguf_ctx())[0];
        }


        void load_hparams();

        void load_vocab();

        bool load_tensors();

    public:
        std::string _name;
        std::string _arch_name;

        tff::core::model::ModelType _type;
        tff::core::model::ModelArchitecture _architecture;

        tff::core::model::ModelHeadParams _head_params;
        tff::core::model::LLMVocabulary _vocabulary;

        std::unordered_map<std::string, std::shared_ptr<tff::core::memory::Tensor> > _tensor_map;

        tff::core::model::ModelParams _model_params;

        std::unordered_map<std::string, std::string> _model_meta_kv;

        std::vector<std::shared_ptr<tff::core::device::DeviceBaseObject> > _devices;

        std::shared_ptr<tff::core::model::LLMModelLoader> _model_loader;
    };
}
#endif //TFFINFER_LLMMODEL_H
