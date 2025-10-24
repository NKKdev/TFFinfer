//
// Created by nkk on 2025/9/28.
//

#include "DeviceCUDA.h"
#include "cudaInc.h"
#include "Logger.h"
namespace tff::core::device::cuda {
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
        size_t freeMem, totalMem;
        cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);
        if (err != cudaSuccess) {
            tff::log::Logger::error("Error: %s" ,cudaGetErrorString(err));
            return ;
        }
    }

    tff::core::device::DeviceType DeviceCUDA::get_device_type(size_t _device_id) {
        return tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU;
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

    void DeviceCUDA::device_init(size_t _device_id) {
        CudaSafeCall(cudaSetDevice(_device_id));
    }

    std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> &DeviceCUDA::get_device_buffer_allocator() {
        std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> allocator = nullptr;

        return allocator;
    }
}
