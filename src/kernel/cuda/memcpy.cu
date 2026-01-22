//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMMemManager.h"

namespace tff::kernel {
    template<typename T>
    static void memcpy_kernel_cuda(
        const std::shared_ptr<tff::core::memory::Tensor> &src,
        const std::shared_ptr<tff::core::memory::Tensor> &dst,
        std::shared_ptr<core::runtime::LLMMemManager> &mem_buffer_manager_ptr,
        tff::core::memory::MemCpyKind kind,
        std::shared_ptr<core::device::DeviceStream> &stream,
        std::shared_ptr<core::device::DeviceEvent> &event,
        std::vector<std::shared_ptr<core::device::DeviceEvent>> &wait_event_list) {
        auto allocator = dst->get_allocator();
        if (allocator == nullptr) {
            tff::log::Logger::error("kernel (%s) allocator is invalid!");
            return;
        }
        for (auto &wait_event : wait_event_list) {
            if (wait_event == nullptr) {
                continue;
            }
            stream->wait_event(wait_event->get_native_event());
        }
        allocator->memcpy_async(src->get_buffer()->ptr(), dst->get_buffer()->ptr(), dst->get_bytes(), kind,
                                stream->get_native_stream());
    }

    static bool is_same_shape(std::array<int64_t, MAX_TENSOR_DIM> &shape1,
                              std::array<int64_t, MAX_TENSOR_DIM> &shape2) {
        for (int i = 0; i < shape1.size(); i++) {
            if (shape1[i] != shape2[i]) {
                return false;
            }
        }
        return true;
    }

    //
    template<typename T>
    void tff::kernel::MemCpy<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), MemCpy<T>::get_op_name().c_str());
        const auto memcpy_kind = get_param_value<tff::core::memory::MemCpyKind>(1, para_ptr);
        auto input_tensors = get_param_value<std::set<std::shared_ptr<tff::core::memory::Tensor>,
            core::memory::Tensor::TensorCompare> >(
            2, para_ptr);
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            3, para_ptr);
        std::shared_ptr<core::runtime::LLMMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(4, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream> >(5, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent> >(6, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent>>>(7, para_ptr);
        if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }

        std::vector<std::shared_ptr<core::memory::Tensor> > inputs;
        for (auto &input_tensor: input_tensors) {
            if (input_tensor->get_shape() != output_tensors->get_shape()) {
                continue;
            }
            if (input_tensor == nullptr || output_tensors == nullptr) {
                tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
                return;
            }
            if (input_tensor->get_buffer() == nullptr) {
                tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
                return;
            }
            memcpy_kernel_cuda<T>(*input_tensors.begin(), output_tensors, mem_buffer_manager_ptr,
                                  memcpy_kind, stream, event, event_list);
        }
    }

    template<typename T>
    std::string tff::kernel::MemCpy<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;
        return name;
    }

    template class tff::kernel::MemCpy<float>;
    template class tff::kernel::MemCpy<double>;
    template class tff::kernel::MemCpy<int32_t>;
    template class tff::kernel::MemCpy<int64_t>;
    template class tff::kernel::MemCpy<Q8_0>;
    REGISTER_OP_OBJECT(MemCpy, float);

    REGISTER_OP_OBJECT(MemCpy, double);

    REGISTER_OP_OBJECT(MemCpy, int32_t);

    REGISTER_OP_OBJECT(MemCpy, int64_t);

    REGISTER_OP_OBJECT(MemCpy, Q8_0);
}
