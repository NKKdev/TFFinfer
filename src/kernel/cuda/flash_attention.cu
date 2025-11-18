//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    //
    template<typename T>
    void tff::kernel::FlashAttn<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        //auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node, op:%s compute!",tff::kernel::FlashAttn<T>::get_op_name().c_str());
    }
    template class tff::kernel::FlashAttn<float>;
    template class tff::kernel::FlashAttn<double>;
    template class tff::kernel::FlashAttn<int32_t>;
    template class tff::kernel::FlashAttn<int64_t>;
    REGISTER_OP_OBJECT(FlashAttn, float);
    REGISTER_OP_OBJECT(FlashAttn, double);
    REGISTER_OP_OBJECT(FlashAttn, int32_t);
    REGISTER_OP_OBJECT(FlashAttn, int64_t);
}