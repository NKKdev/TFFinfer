//
// Created by nkk on 2025/11/3.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    template<typename T>
    void tff::kernel::XGemm<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::XGemm<T>::get_op_name().c_str());
    }

    template class tff::kernel::XGemm<float>;
    template class tff::kernel::XGemm<double>;
    REGISTER_OP_OBJECT(XGemm, float);
    REGISTER_OP_OBJECT(XGemm, double);
    //
    template<typename T>
    void tff::kernel::MemCpy<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::MemCpy<T>::get_op_name().c_str());
    }
    template class tff::kernel::MemCpy<float>;
    template class tff::kernel::MemCpy<double>;
    template class tff::kernel::MemCpy<int32_t>;
    template class tff::kernel::MemCpy<int64_t>;
    REGISTER_OP_OBJECT(MemCpy, float);
    REGISTER_OP_OBJECT(MemCpy, double);
    REGISTER_OP_OBJECT(MemCpy, int32_t);
    REGISTER_OP_OBJECT(MemCpy, int64_t);
    //
    template<typename T>
    void tff::kernel::Mul<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::Mul<T>::get_op_name().c_str());
    }
    template class tff::kernel::Mul<float>;
    template class tff::kernel::Mul<double>;
    template class tff::kernel::Mul<int32_t>;
    template class tff::kernel::Mul<int64_t>;
    REGISTER_OP_OBJECT(Mul, float);
    REGISTER_OP_OBJECT(Mul, double);
    REGISTER_OP_OBJECT(Mul, int32_t);
    REGISTER_OP_OBJECT(Mul, int64_t);
    //
    template<typename T>
    void tff::kernel::Reshape<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::Reshape<T>::get_op_name().c_str());
    }
    template class tff::kernel::Reshape<float>;
    template class tff::kernel::Reshape<double>;
    template class tff::kernel::Reshape<int32_t>;
    template class tff::kernel::Reshape<int64_t>;
    REGISTER_OP_OBJECT(Reshape, float);
    REGISTER_OP_OBJECT(Reshape, double);
    REGISTER_OP_OBJECT(Reshape, int32_t);
    REGISTER_OP_OBJECT(Reshape, int64_t);
    //
    template<typename T>
    void tff::kernel::Rope<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::Rope<T>::get_op_name().c_str());
    }
    template class tff::kernel::Rope<float>;
    template class tff::kernel::Rope<double>;
    template class tff::kernel::Rope<int32_t>;
    template class tff::kernel::Rope<int64_t>;
    REGISTER_OP_OBJECT(Rope, float);
    REGISTER_OP_OBJECT(Rope, double);
    REGISTER_OP_OBJECT(Rope, int32_t);
    REGISTER_OP_OBJECT(Rope, int64_t);
    //
    template<typename T>
    void tff::kernel::FlashAttn<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::FlashAttn<T>::get_op_name().c_str());
    }
    template class tff::kernel::FlashAttn<float>;
    template class tff::kernel::FlashAttn<double>;
    template class tff::kernel::FlashAttn<int32_t>;
    template class tff::kernel::FlashAttn<int64_t>;
    REGISTER_OP_OBJECT(FlashAttn, float);
    REGISTER_OP_OBJECT(FlashAttn, double);
    REGISTER_OP_OBJECT(FlashAttn, int32_t);
    REGISTER_OP_OBJECT(FlashAttn, int64_t);
    //
    template<typename T>
    void tff::kernel::Add<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::Add<T>::get_op_name().c_str());
    }
    template class tff::kernel::Add<float>;
    template class tff::kernel::Add<double>;
    template class tff::kernel::Add<int32_t>;
    template class tff::kernel::Add<int64_t>;
    REGISTER_OP_OBJECT(Add, float);
    REGISTER_OP_OBJECT(Add, double);
    REGISTER_OP_OBJECT(Add, int32_t);
    REGISTER_OP_OBJECT(Add, int64_t);
    //
    template<typename T>
    void tff::kernel::RMSNorm<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::RMSNorm<T>::get_op_name().c_str());
    }
    template class tff::kernel::RMSNorm<float>;
    template class tff::kernel::RMSNorm<double>;
    template class tff::kernel::RMSNorm<int32_t>;
    template class tff::kernel::RMSNorm<int64_t>;
    REGISTER_OP_OBJECT(RMSNorm, float);
    REGISTER_OP_OBJECT(RMSNorm, double);
    REGISTER_OP_OBJECT(RMSNorm, int32_t);
    REGISTER_OP_OBJECT(RMSNorm, int64_t);
}