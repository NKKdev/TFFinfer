//
// Created by nkk on 2025/11/3.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    template<typename T>
    void tff::kernel::XGemm<T>::compute( base::Layout layout,
         base::Transpose a_transpose,
         base::Transpose b_transpose,
         size_t m,  size_t n,  size_t k,
         float alpha,
         Buffer<T> &a_buffer,size_t a_offset,  size_t a_ld,
         Buffer<T> &b_buffer,size_t b_offset,  size_t b_ld,
         float beta,
         Buffer<T> &c_buffer,size_t c_offset,  size_t c_ld) {
    }

    template class tff::kernel::XGemm<float>;
    template class tff::kernel::XGemm<double>;
    REGISTER_OP_OBJECT(XGemm, float);
    REGISTER_OP_OBJECT(XGemm, double);
}