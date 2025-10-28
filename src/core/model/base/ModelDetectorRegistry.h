//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELDETECTORREGISTRY_H
#define TFFINFER_MODELDETECTORREGISTRY_H
#include "model/BaseDefine.h"
#include "model/base/ModelDetectorBase.h"
#include <iostream>
#include <typeinfo>
namespace tff::core::model {
    class ModelDetectyorRegistry {
    public:
        static ModelDetectyorRegistry &get() {
            static ModelDetectyorRegistry instance;
            return instance;
        }


        // 自动探测并返回合适的读取器
        std::shared_ptr<ModelDetectorBase> find_dector(const std::vector<std::string> &architectures) const {
            auto detectors = tff::factory::ModuleFactory::instance()->create_shared_list<ModelDetectorBase>(MODEL_DETECTOR_TYPE);
            for (auto &detector: detectors) {
                auto detector_ptr = detector.second();
                std::cout << "Created object type: " << typeid(*detector_ptr).name() << std::endl;
                if (detector_ptr) {
                    if (detector_ptr->matches(architectures)){
                        return detector_ptr;
                    }
                }
            }
        }

        std::vector<const char *> get_supported_formats() const {
            std::vector<const char *> names;
            auto detectors = tff::factory::ModuleFactory::instance()->create_shared_list<ModelDetectorBase>(MODEL_DETECTOR_TYPE);
            for (auto &detector: detectors) {
                names.push_back(detector.first.c_str());
            }
            return names;
        }
    };
}
#endif //TFFINFER_MODELDETECTORREGISTRY_H