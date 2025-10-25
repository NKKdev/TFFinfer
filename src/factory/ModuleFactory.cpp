//
// Created by nkk on 2025/10/25.
//
#include "ModuleFactory.h"
namespace tff::factory {
    static std::shared_ptr<ModuleFactory> instance_;
    std::shared_ptr<ModuleFactory> &ModuleFactory::instance() {
        if (!instance_) {
            instance_ = std::shared_ptr<ModuleFactory>(new ModuleFactory);
        }
        return instance_;
    }
}