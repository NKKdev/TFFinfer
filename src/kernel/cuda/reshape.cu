//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    //
    template<typename T>
    void tff::kernel::Reshape<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto x = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            0, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        para_ptr->get_param_count() - 1, para_ptr);
        if (x == nullptr || output_tensor == nullptr) {
            return;
        }
        output_tensor->set_buffer_data(x->get_buffer()->ptr(), x->get_bytes(),
            x->get_external_memory_index());
    }

    template class tff::kernel::Reshape<float>;
    template class tff::kernel::Reshape<half>;
    template class tff::kernel::Reshape<double>;
    template class tff::kernel::Reshape<int32_t>;
    template class tff::kernel::Reshape<int64_t>;
    template class tff::kernel::Reshape<Q8_0>;
    REGISTER_OP_OBJECT(Reshape, float);
    REGISTER_OP_OBJECT(Reshape, half);
    REGISTER_OP_OBJECT(Reshape, double);
    REGISTER_OP_OBJECT(Reshape, int32_t);
    REGISTER_OP_OBJECT(Reshape, int64_t);
    REGISTER_OP_OBJECT(Reshape, Q8_0);
}