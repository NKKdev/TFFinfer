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
    void tff::kernel::ViewOP<T, core::device::GPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        return;
    }
    template class tff::kernel::ViewOP<float, core::device::GPUTag>;
    template class tff::kernel::ViewOP<Q8_0, core::device::GPUTag>;
    template class tff::kernel::ViewOP<Q8_0_ALIGNED, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(ViewOP, float, core::device::GPUTag);
    REGISTER_OP_OBJECT_DEVICE(ViewOP, Q8_0, core::device::GPUTag);
    REGISTER_OP_OBJECT_DEVICE(ViewOP, Q8_0_ALIGNED, core::device::GPUTag);
}