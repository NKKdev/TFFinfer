//
// Created by nkk on 2025/10/28.
//

#include "DeviceCPU.h"
#include "FunctionFactory.h"
#include "global/ModelGlobalVar.h"

namespace tff::core::device::cpu {
    void DeviceCPU::get_device_id(std::vector<int> &_device_list) {
    }

    const char *DeviceCPU::get_device_name(size_t _device_id) {
    }

    const char *DeviceCPU::get_device_description(size_t _device_id) {
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

    //
    std::function<tff::kernel::base::OP_CALLBACK_TYPE> DeviceCPU::get_op_func(
        const tff::core::graph::TffOpType &op_type) {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return nullptr;
        }
        std::string op_name = std::string(it->second) + DEVICE_BACKEND_TYPE_CPU;

        return tff::factory::FunctionFactory::instance()->get_callback<tff::kernel::base::OP_CALLBACK_TYPE>(OP_NODE_FLAG,
                                                                          op_name);
    }
}
