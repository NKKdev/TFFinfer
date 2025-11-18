//
// Created by nkk on 2025/11/3.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    template<typename T>
    void tff::kernel::XGemm<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        //auto name = para_ptr->get_param<std::string>(0);
        //tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::XGemm<T>::get_op_name().c_str());
    }

    template class tff::kernel::XGemm<float>;
    template class tff::kernel::XGemm<double>;
    REGISTER_OP_OBJECT(XGemm, float);
    REGISTER_OP_OBJECT(XGemm, double);







}