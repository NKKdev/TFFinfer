//
// Created by nkk on 2026/1/20.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMMemManager.h"

namespace tff::kernel {
    template<typename T>
    void tff::kernel::MemRef<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), MemRef<T>::get_op_name().c_str());
        auto input_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            1, para_ptr);
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
           std::shared_ptr<
               tff::core::runtime::LLMMemManager> >(3, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream>>(4, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent>>(5, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent>>>(6, para_ptr);
        if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }
        if (output_tensors == nullptr) {
            tff::log::Logger::error("op (%s) output tensors is null", name.c_str());
            return;
        }
        if (output_tensors->get_buffer() == nullptr) {
            tff::log::Logger::error("op (%s) output tensors buffer is null", name.c_str());
            return;
        }
        if (input_tensors == nullptr) {
            return;
        }
        *output_tensors = *input_tensors;
    }
    template<typename T>
    std::string tff::kernel::MemRef<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MEM_REF);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();
        return name;
    }

    template class tff::kernel::MemRef<float>;
    template class tff::kernel::MemRef<double>;
    template class tff::kernel::MemRef<int32_t>;
    template class tff::kernel::MemRef<Q8_0>;
    REGISTER_OP_OBJECT(MemRef, float);
    REGISTER_OP_OBJECT(MemRef, double);
    REGISTER_OP_OBJECT(MemRef, int32_t);
    REGISTER_OP_OBJECT(MemRef, Q8_0);
}
