//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_DEVICEBASEOBJECT_H
#define TFFINFER_DEVICEBASEOBJECT_H
#include "ModuleObject.h"
#include "BaseDefine.h"
#include "../mem/MemBufferAllocatorBaseObject.h"
#include <memory>

namespace tff::core::device {
    class DEEP_TFF_API DeviceBaseObject : public tff::module::ModuleObject {
    public:
        DeviceBaseObject();

        ~DeviceBaseObject() override;

    public:
        virtual void get_device_id(std::vector<int> &_device_list) = 0;

        virtual const char *get_device_name(size_t _device_id) = 0;

        virtual const char *get_device_description(size_t _device_id) = 0;

        virtual void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) = 0;

        virtual tff::core::device::DeviceType get_device_type(size_t _device_id) = 0;

        virtual void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) = 0;

        virtual void device_init(size_t _device_id) = 0;

        virtual std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> &get_device_buffer_allocator() = 0;
    };
}


#endif //TFFINFER_DEVICEBASEOBJECT_H
