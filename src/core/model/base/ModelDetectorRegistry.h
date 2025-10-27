//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELDETECTORREGISTRY_H
#define TFFINFER_MODELDETECTORREGISTRY_H
#include "model/BaseDefine.h"
#include "model/base/ModelDetectorBase.h"
namespace tff::core::model {
    class ModelDetectyorRegistry {
    public:
        static ModelDetectyorRegistry &get() {
            static ModelDetectyorRegistry instance;
            return instance;
        }


        // 自动探测并返回合适的读取器
        std::shared_ptr<ModelDetectorBase> &find_dector(const std::vector<std::string> &architectures) const {
            auto detectors = tff::factory::ModuleFactory::instance()->create_shared_list(MODEL_DETECTOR_TYPE);
            for (auto &detector: detectors) {
                auto detector_ptr = std::dynamic_pointer_cast<ModelDetectorBase>(detector.second.front()());
                if (detector_ptr) {
                    if (detector_ptr->matches(architectures)){
                        return detector_ptr;
                    }
                }
            }
        }

        // 获取所有支持的格式（用于日志）
        std::vector<const char *> get_supported_formats() const {
            std::vector<const char *> names;
            auto detectors = tff::factory::ModuleFactory::instance()->create_shared_list(MODEL_DETECTOR_TYPE);
            for (auto &detector: detectors) {
                names.push_back(detector.first.c_str());
            }
            return names;
        }
    };
}
#endif //TFFINFER_MODELDETECTORREGISTRY_H