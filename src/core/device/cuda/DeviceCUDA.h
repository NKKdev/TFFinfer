//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_DEVICECUDA_H
#define TFFINFER_DEVICECUDA_H
#include "../DeviceBaseObject.h"
#include "ModuleFactory.h"
#include "global/GlobalDefine.h"
#include <vector>
namespace tff::core::device::cuda {
    class DeviceCUDA final : public DeviceBaseObject {
    public:
        DeviceCUDA() {
            this->_sched_priority = TFF_DEVICE_PRIORITY_GPU;
        };
        ~DeviceCUDA() override= default;
    public:
        void get_device_id(std::vector<int> &_device_list) override;
        const char *get_device_name(size_t _device_id) override ;
        const char *get_device_description(size_t _device_id) override;
        void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) override;
        tff::core::device::DeviceType get_device_type(size_t _device_id) override ;
        void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) override ;
        void device_init(size_t _device_id) override ;
        std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> get_device_buffer_allocator() override;
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> get_op_func(
            const tff::core::graph::TffOpType &op_type) override ;
    };

}


#endif //TFFINFER_DEVICECUDA_H
