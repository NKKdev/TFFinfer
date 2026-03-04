//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T>
    class Reshape<T,
                core::device::GPUTag> : public base::OPCreatorBase<Reshape<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_RESHAPE;
        }
    };

    //
    template<typename T>
    void tff::kernel::Reshape<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto x = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            ReshapeBuilder::Params::In, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            ReshapeBuilder::Params::Out, para_ptr);
        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            kernel::builder::OpParamBuilderBase<ReshapeBuilder>::CommonParams::Stream, para_ptr);
        if (x == nullptr || output_tensor == nullptr) {
            return;
        }
        output_tensor = x;
    }

    template class tff::kernel::Reshape<float, core::device::GPUTag>;
    template class tff::kernel::Reshape<half, core::device::GPUTag>;
    template class tff::kernel::Reshape<double, core::device::GPUTag>;
    template class tff::kernel::Reshape<int32_t, core::device::GPUTag>;
    template class tff::kernel::Reshape<int64_t, core::device::GPUTag>;
    template class tff::kernel::Reshape<Q8_0, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(Reshape, float, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Reshape, half, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Reshape, double, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Reshape, int32_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Reshape, int64_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Reshape, Q8_0, core::device::GPUTag);
}
