//
// Created by nkk on 2025/9/28.
//

#include "DeviceCUDA.h"
#include "cudaInc.h"
#include "Logger.h"
#include "global/ModelGlobalVar.h"
#include "mem/MemBufferAllocatorCUDA.h"
namespace tff::core::device::cuda {
    REGISTER_MODULE_OBJECT(DeviceCUDA, DeviceBaseObject, DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CUDA)

    void DeviceCUDA::get_device_id(std::vector<int> &_device_list) {
        int device_cnt = 0;
        CudaSafeCall(cudaGetDeviceCount(&device_cnt));
        for (size_t i = 0; i < device_cnt; i++) {
            cudaDeviceProp device_prop{};
            CudaSafeCall(cudaGetDeviceProperties(&device_prop, i));
            _device_list.push_back(device_prop.pciDeviceID);
        }
    }

    const char *DeviceCUDA::get_device_name(size_t _device_id) {
        cudaDeviceProp device_prop{};
        CudaSafeCall(cudaGetDeviceProperties(&device_prop, _device_id));
        return device_prop.name;
    }

    const char *DeviceCUDA::get_device_description(size_t _device_id) {
        cudaDeviceProp device_prop{};
        CudaSafeCall(cudaGetDeviceProperties(&device_prop, _device_id));
        return device_prop.name;
    }

    void DeviceCUDA::get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) {
        CudaSafeCall(cudaMemGetInfo(_free_mem, _total_mem));
    }

    tff::core::device::DeviceType DeviceCUDA::get_device_type(size_t _device_id) {
        return tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU;
    }

    tff::core::device::DeviceType DeviceCUDA::device_type() {
        std::vector<int> device_list;
        get_device_id(device_list);
        if (device_list.empty()) {
            return tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_UNKNOWN;
        }
        return get_device_type(device_list[0]);
    }

    std::string DeviceCUDA::get_device_type_flag(size_t _device_id) {
        return DEVICE_BACKEND_TYPE_CUDA;
    }

    void DeviceCUDA::get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) {
        cudaDeviceProp device_prop{};
        CudaSafeCall(cudaGetDeviceProperties(&device_prop, _device_id));
        _device_props.name = device_prop.name;
        _device_props.caps.async = device_prop.asyncEngineCount > 0 ? true : false;
        _device_props.caps.host_buffer = true;
        _device_props.caps.buffer_from_host_ptr = device_prop.canMapHostMemory != 0;
        _device_props.caps.events = true;
        _device_props.description = device_prop.name;
        _device_props.device_id = device_prop.luid;
        this->get_device_mem(_device_id, &_device_props.memory_free, &_device_props.memory_total);
        _device_props.type = get_device_type(_device_id);
    }

    void DeviceCUDA::device_init() {
        std::vector<int> device_list;
        get_device_id(device_list);
        for (size_t i = 0; i < device_list.size(); i++) {
            CudaSafeCall(cudaSetDevice(device_list[i]));
            auto mem_buffer_allocator = std::make_shared<core::memory::MemBufferAllocatorCUDA>();
            this->_mem_buffer_allocators.insert(std::make_pair(device_list[i], mem_buffer_allocator));
        }

    }

    std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> DeviceCUDA::get_device_buffer_allocator(const int &device_id) {
        return tff::factory::ModuleFactory::instance()->create_shared<
            tff::core::memory::MemBufferAllocatorBaseObject>(MEMORY_ALLOCATOR_FLAG,
                                                             tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
    }

    //
    std::function<tff::kernel::base::OP_CALLBACK_TYPE> DeviceCUDA::get_op_func(
        const tff::core::graph::TffOpType &op_type, const tff::core::memory::DataType &data_type) {
        auto it = core::global::TFF_OP_TYPE_MAP.find(op_type);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return nullptr;
        }
        std::string op_name = std::string(it->second) + std::string("_") + DEVICE_BACKEND_TYPE_CUDA;
        switch (data_type) {
            case tff::core::memory::DataType::TFF_DATA_TYPE_F32:
                op_name = op_name + + tff::core::global::get_type_suffix<float>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F64:
                op_name = op_name + +tff::core::global::get_type_suffix<double>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I32:
                op_name = op_name + +tff::core::global::get_type_suffix<int32_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I64:
                op_name = op_name + +tff::core::global::get_type_suffix<int64_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_I8:
                op_name = op_name + +tff::core::global::get_type_suffix<uint8_t>();
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F16:
                op_name = op_name + +tff::core::global::get_type_suffix<half>();
                break;
            case tff::core::memory::TFF_DATA_TYPE_Q8_0:
                op_name = op_name + + tff::core::global::get_type_suffix<Q8_0>();
                break;
            default:
                break;
        }

        return tff::factory::FunctionFactory::instance()->get_callback<tff::kernel::base::OP_CALLBACK_TYPE>(
            OP_NODE_FLAG,
            op_name);
    }
}
