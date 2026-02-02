//
// Created by nkk on 2025/11/14.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/MemManager.h"

namespace tff::kernel {
    template<typename T>
    static void mem_map2cpu_kernel_cpu(const size_t &model_file_index,
                                       const size_t &offset,
                                       const double &data_size,
                                       const std::shared_ptr<tff::core::model::ModelLoaderBase> &model_loader_ptr,
                                       std::vector<std::shared_ptr<tff::core::memory::Tensor>> &inputs,
                                       std::shared_ptr<tff::core::memory::Tensor> &outputs) {
        if (model_file_index < 0) {
            tff::log::Logger::error("Model file index is invalid");
            return;
        }
        if (data_size < 0) {
            tff::log::Logger::error("Data size is invalid");
            return;
        }
        if (offset < 0) {
            tff::log::Logger::error("Offset is invalid");
            return;
        }

        auto file_map_ptr = model_loader_ptr->get_file_map(model_file_index);
        if (file_map_ptr == nullptr) {
            tff::log::Logger::error("Failed to get file map");
            return;
        }
        auto buffer = (uint8_t *) file_map_ptr->addr() + offset;
        auto allocator = outputs->get_allocator();
        if (allocator == nullptr) {
            tff::log::Logger::error("Failed to create device buffer allocator");
            return;
        }

        allocator->memcopy((void*)buffer, outputs->get_buffer()->ptr(), data_size);
    }

    template<typename T>
    void tff::kernel::MemMap2Cpu<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto model_file_index = kernel::base::get_param_value<size_t>(0, para_ptr);
        const auto offset = kernel::base::get_param_value<size_t>(1, para_ptr);
        const auto data_size = kernel::base::get_param_value<double>(2, para_ptr);
        const auto model_loader_ptr = kernel::base::get_param_value<std::shared_ptr<tff::core::model::ModelLoaderBase> >(3, para_ptr);
        auto input_tensors = kernel::base::get_param_value<std::set<std::shared_ptr<tff::core::memory::Tensor>,
            core::memory::Tensor::TensorCompare> >(
            4, para_ptr);
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            5, para_ptr);

        std::vector<std::shared_ptr<core::memory::Tensor>> inputs;
        for (auto &input_tensor : input_tensors) {
            inputs.push_back(input_tensor);
        }
        mem_map2cpu_kernel_cpu<T>(model_file_index, offset, data_size, model_loader_ptr, inputs, output_tensors);
    }

    template class tff::kernel::MemMap2Cpu<float>;
    template class tff::kernel::MemMap2Cpu<double>;
    template class tff::kernel::MemMap2Cpu<int32_t>;
    template class tff::kernel::MemMap2Cpu<Q8_0>;
    REGISTER_OP_OBJECT(MemMap2Cpu, float);
    REGISTER_OP_OBJECT(MemMap2Cpu, double);
    REGISTER_OP_OBJECT(MemMap2Cpu, int32_t);
    REGISTER_OP_OBJECT(MemMap2Cpu, Q8_0);
}
