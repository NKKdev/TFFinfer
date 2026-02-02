//
// Created by nkk on 2025/11/19.
//
#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/MemManager.h"

namespace tff::kernel {
    template<typename T>
    void tff::kernel::MemRef<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            0, para_ptr);
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        if (output_tensors == nullptr) {
            return;
        }
        if (output_tensors->get_buffer() == nullptr) {
            return;
        }
        if (input_tensor == nullptr) {
            return;
        }
        output_tensors = input_tensor;
   }


    // template class tff::kernel::MemRef<float>;
    // template class tff::kernel::MemRef<double>;
    // template class tff::kernel::MemRef<int32_t>;
    // template class tff::kernel::MemRef<Q8_0>;
    // REGISTER_OP_OBJECT(MemRef, float);
    // REGISTER_OP_OBJECT(MemRef, double);
    // REGISTER_OP_OBJECT(MemRef, int32_t);
    // REGISTER_OP_OBJECT(MemRef, Q8_0);
}
