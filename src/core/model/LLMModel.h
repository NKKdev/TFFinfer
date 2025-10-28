//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_LLMMODEL_H
#define TFFINFER_LLMMODEL_H
#include <unordered_map>
#include <memory>
#include "BaseDefine.h"
#include "LLMVocabulary.h"
#include "mem/BaseDefine.h"
#include "device/DeviceBaseObject.h"
#include "model/base/ModelLoaderBase.h"
#include "ModuleFactory.h"
#include "ModelGlobalVar.h"
#include "graph/GraphNode.h"

namespace tff::core::model {
    class LLMModel {
    public:
        LLMModel() : _type(), _architecture() {
            auto device_list = tff::factory::ModuleFactory::instance()->create_shared_list<
                tff::core::device::DeviceBaseObject>(DEVICE_BACKEND_FLAG);
            auto gpu_device_list = device_list[DEVICE_BACKEND_TYPE_CUDA];
            this->_devices.push_back(std::dynamic_pointer_cast<tff::core::device::DeviceBaseObject>(gpu_device_list()));
            //
            auto cpu_device_list = device_list[DEVICE_BACKEND_TYPE_CPU];
            this->_devices.push_back(std::dynamic_pointer_cast<tff::core::device::DeviceBaseObject>(cpu_device_list()));
        }

        ~LLMModel() = default;

    public:
        //
        bool load_model(const std::vector<std::string> &model_files_path,
                        const tff::core::model::ModelConfig &params);

        //
        bool load_model_config(const std::string &model_config_file_path, tff::core::model::ModelConfig &params);

    protected:
        void load_stats();

        inline void load_arch() {
            _arch_name = get_value<tff::core::model::ModelMetaKV,
                ModelContext::BasicType, std::string>(tff::core::model::ModelMetaKV::LLM_KV_GENERAL_ARCHITECTURE,
                                                      _model_loader->get_model_context())[0];
        }


        void load_hparams();

        void load_vocab() const;

        bool load_layers();

    public:
        std::string _name;
        std::string _arch_name;
        uint64_t _n_elements = 0;
        size_t _n_bytes = 0;

        tff::core::model::ModelType _type;
        tff::core::model::ModelArchitectureType _architecture;

        tff::core::model::ModelConfig _model_config;
        std::unique_ptr<tff::core::model::LLMLLaMaVocabulary> _vocabulary_ptr;

        //std::unordered_map<std::string, std::shared_ptr<tff::core::memory::Tensor> > _tensor_map;


        std::unordered_map<std::string, std::string> _model_meta_kv;

        std::vector<std::shared_ptr<tff::core::device::DeviceBaseObject> > _devices;

        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;
        //
        std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<tff::core::graph::GraphNode> >
        _layer_map;
    };
}
#endif //TFFINFER_LLMMODEL_H
