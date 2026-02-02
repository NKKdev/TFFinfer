//
// Created by nkk on 2025/11/18.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    //
    template<typename T>
    void tff::kernel::Add<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        //auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node op:%s compute!",tff::kernel::Add<T>::get_op_name().c_str());
    }

    template class tff::kernel::Add<float>;
    template class tff::kernel::Add<double>;
    template class tff::kernel::Add<int32_t>;
    template class tff::kernel::Add<int64_t>;
    template class tff::kernel::Add<Q8_0>;
    REGISTER_OP_OBJECT(Add, float);
    REGISTER_OP_OBJECT(Add, double);
    REGISTER_OP_OBJECT(Add, int32_t);
    REGISTER_OP_OBJECT(Add, int64_t);
    REGISTER_OP_OBJECT(Add, Q8_0);
}