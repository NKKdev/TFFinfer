//
// Created by nkk on 2025/10/26.
//

#ifndef TFFINFER_GLOBALDEFINE_H
#define TFFINFER_GLOBALDEFINE_H
#include "ModuleFactory.h"
#include "device/DeviceBaseObject.h"
#include "OPDefine.h"
namespace tff::core::global {
#define TFF_MAX_OP_PARAMS      64
#define MODEL_DETECTOR_TYPE "ModelDetector"
#define MODEL_LOADER_TYPE "ModelLoader"
#define MODEL_READER_TYPE "ModelReader"
#define MEMORY_ALLOCATOR_FLAG "MEMORY"

#define DEVICE_BACKEND_FLAG "DEVICE"
#define DEVICE_BACKEND_TYPE_CUDA "CUDA"
#define DEVICE_BACKEND_TYPE_CPU "CPU"
#define BUFFER_ALLOCATOR_FLAG "BufferAllocator"
    //
#define CREATE_LAYER_FLAG "CREATE_LAYER"
#define BUILD_GRAPH_FLAG "BUILD_GRAPH"
    //task graph
#define TASK_GRAPH_FLAG "TASK_GRAPH"
#define TASK_GRAPH_TYPE "TASK_SCHEDULER"
    //TOKENIZER
#define TOKENIZER_FLAG "TOKENIZER"
    //
#define MAX_BATCH_SIZE 512
    //OP
#define OP_NODE_FLAG "OP_NODE"

    static size_t get_device_size(const std::string &device_key) {
        auto devices = tff::factory::ModuleFactory::instance()->create_shared_list<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG);
        int n_device_num = 0;
        for (const auto& [key, info] : devices) {
            if (tff::factory::ModuleKeyType(device_key) != tff::factory::ModuleKeyType(key)) {
                continue;
            }
            std::vector<int> device_list;
            info.creator()->get_device_id(device_list);
            n_device_num = device_list.size();
        }
        return n_device_num;
    }
    //

}
#endif //TFFINFER_GLOBALDEFINE_H
