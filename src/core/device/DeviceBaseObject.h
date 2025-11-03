//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_DEVICEBASEOBJECT_H
#define TFFINFER_DEVICEBASEOBJECT_H
#include <memory>
#include "ModuleObject.h"
#include "BaseDefine.h"
#include "mem/MemBufferAllocatorBaseObject.h"


namespace tff::core::device {

    class DEEP_TFF_API DeviceBaseObject : public tff::module::ModuleObject {
    public:
        DeviceBaseObject() = default;

        ~DeviceBaseObject() override = default;

    public:
        virtual void get_device_id(std::vector<int> &_device_list) = 0;

        virtual const char *get_device_name(size_t _device_id) = 0;

        virtual const char *get_device_description(size_t _device_id) = 0;

        virtual void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) = 0;

        virtual tff::core::device::DeviceType get_device_type(size_t _device_id) = 0;

        virtual void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) = 0;

        virtual void device_init(size_t _device_id) = 0;

        virtual std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> &get_device_buffer_allocator() = 0;
    public:
        uint32_t _sched_priority = TFF_DEVICE_PRIORITY_DEFAULT;
    };

    // 优先级排序;
    struct DevicePtrComparator {
        bool operator()(
            const std::shared_ptr<tff::core::device::DeviceBaseObject>& a,
            const std::shared_ptr<tff::core::device::DeviceBaseObject>& b
        ) const {
            if (!a && !b) return false;
            if (!a) return true;
            if (!b) return false;
            return a->_sched_priority > b->_sched_priority;  // 字典序降序
        }
    };
}


#endif //TFFINFER_DEVICEBASEOBJECT_H
