//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_DEVICEBASEOBJECT_H
#define TFFINFER_DEVICEBASEOBJECT_H
#include <memory>
#include "ModuleObject.h"
#include "BaseDefine.h"
#include "mem/MemBufferAllocatorBaseObject.h"
#include "graph/BaseDefine.h"
#include "kernel/include/TFFOPCreatorBase.h"

namespace tff::core::device {
    class DEEP_TFF_API DeviceStream {
    public:
        virtual ~DeviceStream() = default;

        virtual void synchronize() = 0;

        virtual void *get_native_stream() = 0;

        virtual void wait_event(void *event_handle) = 0;

        [[nodiscard]] virtual bool is_valid() const = 0;
    };

    //
    class DEEP_TFF_API DeviceEvent {
    public:
        virtual ~DeviceEvent() = default;

        virtual void record(const std::shared_ptr<DeviceStream> &stream) = 0;

        virtual void *get_native_event() = 0;

        [[nodiscard]] virtual bool is_valid() const = 0;
    };


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

        virtual tff::core::device::DeviceType device_type() = 0;

        //
        virtual std::string get_device_type_flag(size_t _device_id) = 0;

        virtual void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) = 0;

        virtual void device_init() = 0;

        virtual std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> get_device_buffer_allocator(
            const int &device_id) = 0;

        //
        virtual std::function<tff::kernel::base::OP_CALLBACK_TYPE> get_op_func(
            const tff::core::graph::TffOpType &op_type, const tff::core::memory::DataType &data_type) = 0;

        //
        virtual std::shared_ptr<DeviceStream> create_stream(int device_id) = 0;

        virtual std::shared_ptr<DeviceEvent> create_event(int device_id) = 0;

        //
        virtual float elapsed_time(
            const std::shared_ptr<DeviceEvent> &start,
            const std::shared_ptr<DeviceEvent> &stop
        ) = 0;

    public:
        uint32_t _sched_priority = TFF_DEVICE_PRIORITY_GPU;
        std::unordered_map<int, std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> >
        _mem_buffer_allocators;
    };

    // 优先级排序;
    struct DeviceComparator {
        bool operator()(
            const std::shared_ptr<tff::core::device::DeviceBaseObject> &a,
            const std::shared_ptr<tff::core::device::DeviceBaseObject> &b
        ) const {
            if (!a && !b) return false;
            if (!a) return true;
            if (!b) return false;
            return a->_sched_priority > b->_sched_priority;
        }
    };

    static size_t get_device_size(const std::string &device_key) {
        auto devices = tff::factory::ModuleFactory::instance()->create_shared_list<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG);
        int n_device_num = 0;
        for (const auto &[key, info]: devices) {
            if (tff::factory::ModuleKeyType(device_key) != tff::factory::ModuleKeyType(key)) {
                continue;
            }
            std::vector<int> device_list;
            info.creator()->get_device_id(device_list);
            n_device_num = device_list.size();
        }
        return n_device_num;
    }
}


#endif //TFFINFER_DEVICEBASEOBJECT_H
