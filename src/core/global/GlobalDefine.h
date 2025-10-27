//
// Created by nkk on 2025/10/26.
//

#ifndef TFFINFER_GLOBALDEFINE_H
#define TFFINFER_GLOBALDEFINE_H
#include "ModuleFactory.h"
namespace tff::core::global {
#define TFF_MAX_OP_PARAMS      64
#define MODEL_DETECTOR_TYPE "ModelDetector"
#define MODEL_LOADER_TYPE "ModelLoader"
#define MODEL_READER_TYPE "ModelReader"
    static size_t get_device_size(const std::string &device_key) {
        auto devices = tff::factory::ModuleFactory::instance()->create_shared_list("DEVICE");
        return devices[device_key].size();
    }
}
#endif //TFFINFER_GLOBALDEFINE_H