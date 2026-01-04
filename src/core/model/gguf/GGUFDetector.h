//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_LLAMADETECTOR_H
#define TFFINFER_LLAMADETECTOR_H
#include "global/ModelGlobalVar.h"
#include "model/base/ModelLoaderBase.h"
#include "model/base/ModelDetectorBase.h"
#include "model/BaseDefine.h"
#include "global/GlobalDefine.h"
#include "../llama/LLAMACreator.h"

namespace tff::core::model {
    class GGUFDetector : public ModelDetectorBase {
    public:
        GGUFDetector() = default;

        ~GGUFDetector() override = default;

    public:
        // 检查模型是否匹配此架构
        [[nodiscard]] bool matches(const std::string &file_format) const override {
            return file_format == std::string(
                       tff::core::model::to_string(tff::core::model::ModelFileFormat::TFF_MODEL_FORMAT_GGUF));
        }

        // 获取该模型文件格式的名称
        [[nodiscard]] const char *name() const override {
            return tff::core::model::to_string(tff::core::model::ModelFileFormat::TFF_MODEL_FORMAT_GGUF);
        }

        // 创建该模型的加载器
        std::shared_ptr<ModelLoaderBase> create_loader() override {
            return tff::factory::ModuleFactory::instance()->create_shared<ModelLoaderBase>(MODEL_LOADER_FLAG,
                std::string(
                    tff::core::model::to_string(tff::core::model::ModelFileFormat::TFF_MODEL_FORMAT_GGUF)));
        }
    };
}


#endif //TFFINFER_LLAMADETECTOR_H
