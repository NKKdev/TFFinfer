//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMWeightMemManager.h"
namespace tff::kernel {
    //
    template<typename T>
    void tff::kernel::MemCpy<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!",name.c_str(),MemCpy<T>::get_op_name().c_str());

        const auto model_file_index = get_param_value<uint16_t>(1, para_ptr);
        const auto offset = get_param_value<size_t>(2, para_ptr);
        const auto data_size = get_param_value<double>(3, para_ptr);
        const auto model_loader_ptr = get_param_value<std::shared_ptr<tff::core::model::ModelLoaderBase> >(4, para_ptr);
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            5, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            6, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(7, para_ptr);
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