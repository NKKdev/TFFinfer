//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_LLAMADETECTOR_H
#define TFFINFER_LLAMADETECTOR_H
#include "model/ModelGlobalVar.h"
#include "model/base/ModelLoaderBase.h"
#include "model/base/ModelDetectorBase.h"
#include "model/BaseDefine.h"
#include "global/GlobalDefine.h"
#include "LLAMACreator.h"
namespace tff::core::model {
    class LLAMADetector : public ModelDetectorBase {
    public:
        LLAMADetector() = default;

        ~LLAMADetector() override = default;

    public:
        // 检查模型是否匹配此架构
        bool matches(const std::vector<std::string> &architectures) const override {
            return std::find(architectures.begin(), architectures.end(), "LlamaForCausalLM") != architectures.end();
        }

        // 获取该模型的名称（用于日志）
        const char *name() const override {
            return LLM_ARCH_NAMES.find(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA)->second;
        }
        tff::core::model::ModelArchitectureType arch() const override {
            return tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA;
        }
        // 创建该模型的加载器
        std::shared_ptr<ModelLoaderBase> create_loader() override {
            return tff::factory::ModuleFactory::instance()->create_shared<ModelLoaderBase>(MODEL_LOADER_TYPE,
                                                                       std::string(
                                                                           LLM_ARCH_NAMES.find(
                                                                               tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA)
                                                                           ->second));
        }
        //
        void model_registry() override  {
            LLAMACreator::registry_function();
        }

        // （可选）优先级，用于解决冲突
        int priority() const override { return 50; } // 默认优先级
    };

    REGISTER_MODULE_OBJECT(LLAMADetector, ModelDetectorBase,MODEL_DETECTOR_TYPE,
                           std::string(LLM_ARCH_NAMES.find(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA
                           )->second));
}


#endif //TFFINFER_LLAMADETECTOR_H
