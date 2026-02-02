//
// Created by nkk on 2026/1/29.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "utils/util.h"

namespace tff::kernel {
        //
    template<typename T>
    void tff::kernel::ViewOP<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        return;
    }
    template class tff::kernel::RMSNorm<float>;
    template class tff::kernel::RMSNorm<Q8_0>;
    template class tff::kernel::RMSNorm<Q8_0_ALIGNED>;
    REGISTER_OP_OBJECT(ViewOP, float);
    REGISTER_OP_OBJECT(ViewOP, Q8_0);
    REGISTER_OP_OBJECT(ViewOP, Q8_0_ALIGNED);
}