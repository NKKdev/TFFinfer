//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMWeightMemManager.h"

namespace tff::kernel {
    template<typename T>
    static void memcpy_kernel_cuda(
        const std::shared_ptr<tff::core::memory::Tensor> &src,
        const std::shared_ptr<tff::core::memory::Tensor> &dst,
        std::shared_ptr<core::runtime::LLMWeightMemManager> &mem_buffer_manager_ptr,
        tff::core::memory::MemCpyKind kind) {
        if (src == nullptr || dst == nullptr) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        if (src->get_buffer() == nullptr) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        auto mem_buffer_pair = mem_buffer_manager_ptr->get_gpu_memory();
        dst->set_buffer_data(mem_buffer_pair.second, dst->get_bytes(), mem_buffer_pair.first);
        dst->get_allocator()->memcpy(src->get_buffer()->ptr(), mem_buffer_pair.second,
        dst->get_bytes(), kind);
        //释放缓存;
        mem_buffer_manager_ptr->reset_cpu_mapped_memory(src->get_external_memory_index());
    }

    //
    template<typename T>
    void tff::kernel::MemCpy<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), MemCpy<T>::get_op_name().c_str());
        const auto memcpy_kind = get_param_value<tff::core::memory::MemCpyKind>(1, para_ptr);
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            2, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            3, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(4, para_ptr);

        if (input_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        if (output_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        auto tmp = *input_tensors.begin();
        memcpy_kernel_cuda<T>(*input_tensors.begin(), *output_tensors.begin(), mem_buffer_manager_ptr,
                              tff::core::memory::MemCpyKind(memcpy_kind));
    }

    template class tff::kernel::MemCpy<float>;
    template class tff::kernel::MemCpy<double>;
    template class tff::kernel::MemCpy<int32_t>;
    template class tff::kernel::MemCpy<int64_t>;
    REGISTER_OP_OBJECT(MemCpy, float);

    REGISTER_OP_OBJECT(MemCpy, double);

    REGISTER_OP_OBJECT(MemCpy, int32_t);

    REGISTER_OP_OBJECT(MemCpy, int64_t);
}
