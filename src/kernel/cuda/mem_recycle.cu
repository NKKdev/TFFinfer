//
// Created by nkk on 2/10/26.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "core/runtime/KVCache.h"
#include "include/kernel_util.h"

namespace tff::kernel {
    template<typename T>
       class MemOptOP<T, core::device::GPUTag> : public base::OPCreatorBase<MemOptOP<T, core::device::GPUTag>, T, core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MEM_RECYCLE;
        }
    };
    template<typename T>
    void tff::kernel::MemOptOP<T, core::device::GPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        // auto input_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
        //    MemOptOPBuilder::Params::In, para_ptr);
        // auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
        //     MemOptOPBuilder::Params::Out, para_ptr);
        // if (input_tensor ==  nullptr || output_tensor == nullptr ||
        //     input_tensor->get_buffer() == nullptr || output_tensor->get_buffer() == nullptr) {
        //     tff::log::Logger::error("MemOptOP::compute: input or output tensor is nullptr.");
        //     return;
        // }
    }


    template class tff::kernel::MemOptOP<Q8_0_ALIGNED, core::device::GPUTag>;
    template class tff::kernel::MemOptOP<Q8_0, core::device::GPUTag>;
    template class tff::kernel::MemOptOP<float, core::device::GPUTag>;
    template class tff::kernel::MemOptOP<half, core::device::GPUTag>;
    template class tff::kernel::MemOptOP<int32_t, core::device::GPUTag>;
    template class tff::kernel::MemOptOP<int64_t, core::device::GPUTag>;

    REGISTER_OP_OBJECT_DEVICE(MemOptOP, Q8_0_ALIGNED, core::device::GPUTag);
    REGISTER_OP_OBJECT_DEVICE(MemOptOP, Q8_0, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemOptOP, float, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemOptOP, half, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemOptOP, int32_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemOptOP, int64_t, core::device::GPUTag);
}
