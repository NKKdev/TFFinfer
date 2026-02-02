//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    //
    template<typename T>
    void tff::kernel::Reshape<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        //auto name = para_ptr->get_param<std::string>(0);
        //tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::Reshape<T>::get_op_name().c_str());
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