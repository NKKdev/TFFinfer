//
// Created by nkk on 2025/10/26.
//

#ifndef TFFINFER_GLOBALDEFINE_H
#define TFFINFER_GLOBALDEFINE_H
#include "ModuleFactory.h"
#include "mem/BaseDefine.h"
#include "device/BaseDefine.h"
#define TFF_MAX_OP_PARAMS      64
#define TASK_FLOW_MANAGER_FLAG "TASK_FLOW_MANAGER"
#define WEIGHT_MEM_BUFFER_MANAGER_FLAG "WEIGHT_MEM_BUFFER_MANAGER"
#define BATCH_MANAGER_FLAG "BATCH_MANAGER"

#define MODEL_CREATOR_FLAG "MODELCreator"
#define MODEL_DETECTOR_FLAG "ModelDetector"
#define MODEL_LOADER_FLAG "ModelLoader"
#define MODEL_READER_FLAG "ModelReader"
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
#define MODEL_HIDDEN_DIM 960
#define MAX_BATCH_SIZE 512
#define MAX_SEQ_LENGTH 8192
#define MAX_PARAM_BUFFER_SIZE 512
#define MAX_PARAM_COUNT 64
    //OP
#define OP_NODE_FLAG "OP_NODE"
//
#define MAX_PREFETCH_BUFFER_SIZE 16
#define MAX_TENSOR_DIM 4
#define MAX_TENSOR_INPUTS 5
namespace tff::core::global {

    //
    static memory::MemCpyKind make_cpy_kind(device::DeviceType &src_type, device::DeviceType &dst_type) {
        if (src_type == device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU && dst_type == device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU) {
            return memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE;
        }else if (src_type == device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU && dst_type == device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU) {
            return memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2DEVICE;
        } else if (src_type == device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU && dst_type == device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU) {
            return memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST;
        }else {
            return memory::MemCpyKind::TFF_MEM_CPY_TYPE_NORMAL;
        }
    }
}
#endif //TFFINFER_GLOBALDEFINE_H
