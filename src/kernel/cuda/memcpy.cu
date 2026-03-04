//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/MemManager.h"

namespace tff::kernel {
    template<typename T>
    static void memcpy_kernel_cuda(
        const std::shared_ptr<tff::core::memory::Tensor> &src,
        const std::shared_ptr<tff::core::memory::Tensor> &dst,
        tff::core::memory::MemCpyKind kind,
        std::shared_ptr<core::device::DeviceStream> &stream) {
        auto allocator = dst->get_allocator();
        if (allocator == nullptr) {
            tff::log::Logger::error("kernel (%s) allocator is invalid!");
            return;
        }

        allocator->memcpy_async(src->get_buffer()->ptr(), dst->get_buffer()->ptr(), dst->get_bytes(), kind,
                                stream->get_native_stream());
    }

    static bool is_same_shape(std::array<int64_t, MAX_TENSOR_DIM> &shape1,
                              std::array<int64_t, MAX_TENSOR_DIM> &shape2) {
        for (int i = 0; i < shape1.size(); i++) {
            if (shape1[i] != shape2[i]) {
                return false;
            }
        }
        return true;
    }

    template<typename T>
    class MemCpy<T,
                core::device::GPUTag> : public base::OPCreatorBase<MemCpy<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MEM_CPY;
        }
    };

    //
    template<typename T>
    void tff::kernel::MemCpy<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto memcpy_kind = kernel::base::get_param_value<tff::core::memory::MemCpyKind>(
            MemCpyBuilder::Params::MemCpyKind, para_ptr);
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            MemCpyBuilder::Params::Out, para_ptr);
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(
            MemCpyBuilder::Params::In, para_ptr);
        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            kernel::builder::OpParamBuilderBase<MemCpyBuilder>::CommonParams::Stream, para_ptr);


        if (input_tensor == nullptr || output_tensors == nullptr) {
            return;;
        }
        if (input_tensor->get_shape() != output_tensors->get_shape()) {
            return;;
        }
        if (input_tensor->get_buffer() == nullptr) {
            return;;
        }
        memcpy_kernel_cuda<T>(input_tensor, output_tensors,
                              memcpy_kind, stream);
    }


    template class tff::kernel::MemCpy<float, core::device::GPUTag>;
    template class tff::kernel::MemCpy<double, core::device::GPUTag>;
    template class tff::kernel::MemCpy<int32_t, core::device::GPUTag>;
    template class tff::kernel::MemCpy<int64_t, core::device::GPUTag>;
    template class tff::kernel::MemCpy<Q8_0, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(MemCpy, float, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemCpy, double, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemCpy, int32_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemCpy, int64_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MemCpy, Q8_0, core::device::GPUTag);
}
