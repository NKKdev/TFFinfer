//
// Created by nkk on 2025/10/28.
//

#include "DeviceCPU.h"
namespace tff::core::device::cpu {
    void DeviceCPU::get_device_id(std::vector<int> &_device_list) {
    }

    const char * DeviceCPU::get_device_name(size_t _device_id) {
    }

    const char * DeviceCPU::get_device_description(size_t _device_id) {
    }

    void DeviceCPU::get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) {
    }

    tff::core::device::DeviceType DeviceCPU::get_device_type(size_t _device_id) {
    }

    void DeviceCPU::get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) {
    }

    void DeviceCPU::device_init(size_t _device_id) {
    }

    std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> DeviceCPU::get_device_buffer_allocator() {
    }
}
