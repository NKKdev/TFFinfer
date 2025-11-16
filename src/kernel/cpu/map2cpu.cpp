//
// Created by nkk on 2025/11/14.
//

#include "include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T>
    void tff::kernel::MemMap2Cpu<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        tff::log::Logger::info("op:%s compute!", tff::kernel::MemMap2Cpu<T>::get_op_name().c_str());
        //auto para1 = para_ptr->get_param<T>(0);
    }

    template class tff::kernel::MemMap2Cpu<float>;
    template class tff::kernel::MemMap2Cpu<double>;
    template class tff::kernel::MemMap2Cpu<int32_t>;
    REGISTER_OP_OBJECT(MemMap2Cpu, float);
    REGISTER_OP_OBJECT(MemMap2Cpu, double);
    REGISTER_OP_OBJECT(MemMap2Cpu, int32_t);
}
