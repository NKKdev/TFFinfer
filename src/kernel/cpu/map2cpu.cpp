//
// Created by nkk on 2025/11/14.
//

#include "include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T>
    void tff::kernel::MemMap2Cpu<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::MemMap2Cpu<T>::get_op_name().c_str());
        //auto para1 = para_ptr->get_param<T>(0);
    }

    template class tff::kernel::MemMap2Cpu<float>;
    template class tff::kernel::MemMap2Cpu<double>;
    template class tff::kernel::MemMap2Cpu<int32_t>;
    REGISTER_OP_OBJECT(MemMap2Cpu, float);

    REGISTER_OP_OBJECT(MemMap2Cpu, double);

    REGISTER_OP_OBJECT(MemMap2Cpu, int32_t);

    //
    template<typename T>
    void tff::kernel::Embedding<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::MemMap2Cpu<T>::get_op_name().c_str());
        //auto para1 = para_ptr->get_param<T>(0);
    }

    template class tff::kernel::Embedding<float>;
    template class tff::kernel::Embedding<double>;
    template class tff::kernel::Embedding<int32_t>;
    REGISTER_OP_OBJECT(Embedding, float);
    REGISTER_OP_OBJECT(Embedding, double);
    REGISTER_OP_OBJECT(Embedding, int32_t);
}
