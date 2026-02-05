//
// Created by nkk on 2026/1/20.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/MemManager.h"

namespace tff::kernel {
    template<typename T>
    void tff::kernel::MemRef<T, core::device::CPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto input_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            0, para_ptr);
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                         para_ptr->get_param_count() - 1, para_ptr);

        if (output_tensors == nullptr) {
            return;
        }
        if (input_tensors == nullptr) {
            return;
        }
        *output_tensors = *input_tensors;
    }


    template class tff::kernel::MemRef<float, core::device::CPUTag>;
    template class tff::kernel::MemRef<double, core::device::CPUTag>;
    template class tff::kernel::MemRef<int32_t, core::device::CPUTag>;
    template class tff::kernel::MemRef<Q8_0_ALIGNED, core::device::CPUTag>;
    template class tff::kernel::MemRef<Q8_0, core::device::CPUTag>;
    REGISTER_OP_OBJECT_DEVICE(MemRef, float, core::device::CPUTag);
    REGISTER_OP_OBJECT_DEVICE(MemRef, double, core::device::CPUTag);
    REGISTER_OP_OBJECT_DEVICE(MemRef, int32_t, core::device::CPUTag);
    REGISTER_OP_OBJECT_DEVICE(MemRef, Q8_0_ALIGNED, core::device::CPUTag);
    REGISTER_OP_OBJECT_DEVICE(MemRef, Q8_0, core::device::CPUTag);
}
