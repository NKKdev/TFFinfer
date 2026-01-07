//
// Created by nkk on 2025/11/14.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMMemManager.h"

namespace tff::kernel {
    template<typename T>
    static void mem_map2cpu_kernel_cpu(const int &model_file_index,
                                       const int &offset,
                                       const double &data_size,
                                       const std::shared_ptr<tff::core::model::ModelLoaderBase> &model_loader_ptr,
                                       std::vector<std::shared_ptr<tff::core::memory::Tensor>> &inputs,
                                       std::vector<std::shared_ptr<tff::core::memory::Tensor>> &outputs,
                                       std::shared_ptr<
                                           tff::core::runtime::LLMMemManager> &mem_buffer_manager_ptr) {
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
        if (inputs.size() != 1) {
            tff::log::Logger::error("Number of input tensors provided is invalid");
            return;
        }
        auto file_map_ptr = model_loader_ptr->get_file_map(model_file_index);
        if (file_map_ptr == nullptr) {
            tff::log::Logger::error("Failed to get file map");
            return;
        }
        auto buffer = (uint8_t *) file_map_ptr->addr() + offset;
        auto input_tensor = *inputs.begin();
        if (input_tensor == nullptr) {
            tff::log::Logger::error("Failed to get input tensor");
            return;
        }
        if (input_tensor->is_allocated()) {
            tff::log::Logger::error("mem_map2cpu_kernel_cpu Input tensor is already allocated");
            return;
        }
        // auto mem_buffer = mem_buffer_manager_ptr->get_cpu_mapped_memory();
        // if (mem_buffer.second == nullptr) {
        //     //tff::log::Logger::error("Failed to get memory mapped memory");
        //     return;
        // }
        // input_tensor->set_buffer_data(mem_buffer.second, input_tensor->get_bytes(), mem_buffer.first);
        auto allocator = input_tensor->get_allocator();
        if (allocator == nullptr) {
            tff::log::Logger::error("Failed to create device buffer allocator");
            return;
        }

        //std::shared_ptr<tff::core::memory::Tensor> output_tensor = input_tensor;
        //allocator->memcpy((void*)buffer, mem_buffer.second, data_size);
    }

    template<typename T>
    void tff::kernel::MemMap2Cpu<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!",name.c_str(),MemMap2Cpu<T>::get_op_name().c_str());

        const auto model_file_index = get_param_value<uint16_t>(1, para_ptr);
        const auto offset = get_param_value<size_t>(2, para_ptr);
        const auto data_size = get_param_value<double>(3, para_ptr);
        const auto model_loader_ptr = get_param_value<std::shared_ptr<tff::core::model::ModelLoaderBase> >(4, para_ptr);
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor>> >(
            5, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor>> >(
            6, para_ptr);
        std::shared_ptr<core::runtime::LLMMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(7, para_ptr);

        mem_map2cpu_kernel_cpu<T>(model_file_index, offset, data_size, model_loader_ptr, input_tensors, output_tensors,
                                  mem_buffer_manager_ptr);
    }
    template<typename T>
    std::string tff::kernel::MemMap2Cpu<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CPU + tff::core::global::get_type_suffix<T>();
        return name;
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
