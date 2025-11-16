//
// Created by nkk on 2025/11/3.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    template<typename T>
    void tff::kernel::XGemm<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
    }

    template class tff::kernel::XGemm<float>;
    template class tff::kernel::XGemm<double>;
    REGISTER_OP_OBJECT(XGemm, float);
    REGISTER_OP_OBJECT(XGemm, double);
    //
    template<typename T>
    void tff::kernel::MemCpy<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        tff::log::Logger::info("op: %s compute!!");
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