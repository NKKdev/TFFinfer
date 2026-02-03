//
// Created by nkk on 2026/2/3.



#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    //
    template<typename T>
    void tff::kernel::ViewOP<T, core::device::CPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        return;
    }
    template class tff::kernel::ViewOP<float, core::device::CPUTag>;
    template class tff::kernel::ViewOP<Q8_0, core::device::CPUTag>;
    template class tff::kernel::ViewOP<Q8_0_ALIGNED, core::device::CPUTag>;
    REGISTER_OP_OBJECT_DEVICE(ViewOP, float, core::device::CPUTag);
    REGISTER_OP_OBJECT_DEVICE(ViewOP, Q8_0, core::device::CPUTag);
    REGISTER_OP_OBJECT_DEVICE(ViewOP, Q8_0_ALIGNED, core::device::CPUTag);
}