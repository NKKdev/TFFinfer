//
// Created by nkk on 2026/1/20.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/MemManager.h"

namespace tff::kernel {
    template<typename T>
    class MemRef<T,
                core::device::GPUTag> : public base::OPCreatorBase<MemRef<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MEM_REF;
        }
    };

    template<typename T>
    void tff::kernel::MemRef<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto input_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            MemRefBuilder::Params::In, para_ptr);
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            MemRefBuilder::Params::Out, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            kernel::builder::OpParamBuilderBase<MemRefBuilder>::CommonParams::Stream, para_ptr);

        if (output_tensors == nullptr) {
            return;
        }
        if (input_tensors == nullptr) {
            return;
        }
        *output_tensors = *input_tensors;
    }


    template class tff::kernel::MemRef<float, core::device::GPUTag>;
    template class tff::kernel::MemRef<double, core::device::GPUTag>;
    template class tff::kernel::MemRef<int32_t, core::device::GPUTag>;
    template class tff::kernel::MemRef<Q8_0_ALIGNED, core::device::GPUTag>;
    template class tff::kernel::MemRef<Q8_0, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(MemRef, float, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemRef, double, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemRef, int32_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemRef, Q8_0_ALIGNED, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemRef, Q8_0, core::device::GPUTag);
}
