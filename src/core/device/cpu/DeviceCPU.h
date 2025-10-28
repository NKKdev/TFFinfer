//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_DEVICECPU_H
#define TFFINFER_DEVICECPU_H
#include "device/DeviceBaseObject.h"
#include "global/GlobalDefine.h"
namespace tff::core::device::cpu {
    class DeviceCPU : public DeviceBaseObject {
    public:
        DeviceCPU() = default;

        ~DeviceCPU() override = default;

    public:
        void get_device_id(std::vector<int> &_device_list) override;
        const char *get_device_name(size_t _device_id) override ;
        const char *get_device_description(size_t _device_id) override;
        void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) override;
        tff::core::device::DeviceType get_device_type(size_t _device_id) override ;
        void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) override ;
        void device_init(size_t _device_id) override ;
        std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> &get_device_buffer_allocator() override;
    };



    REGISTER_MODULE_OBJECT(DeviceCPU, DeviceBaseObject, DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CPU);
}

#endif //TFFINFER_DEVICECPU_H
