//
// Created by nkk on 2026/2/11.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "core/runtime/KVCache.h"
#include "kernel/include/kernel_util.h"
#include "kernel/include/TFFOPCreatorBase.h"
namespace tff::kernel {
    __global__ __forceinline__ void fp32_to_fp16_kernel(half *dst, const float *src, int n) {
        const int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_id >= n / 2) {
            return;
        }
        float2 *src_ptr = (float2*)src;
        half2 *dst_ptr = (half2*)dst;
        dst_ptr[thread_id] = __float22half2_rn(src_ptr[thread_id]);
    }

    template<typename T>
    void fp32_to_fp16(T *dst, const float *src, int n,
        const std::shared_ptr<core::device::DeviceStream> &stream) {
        const int block_size = 256;
        const int grid_size = (n / 2 + block_size - 1) / block_size;
        if constexpr (std::is_same_v<T, half>) {
            fp32_to_fp16_kernel<<<grid_size, block_size, 0,
                              static_cast<cudaStream_t>(stream->get_native_stream())>>>(
                                  dst, src, n);
        }

    }
    template<typename T>
    class ConvertOP<T, core::device::GPUTag> : public kernel::base::OPCreatorBase<ConvertOP<T, core::device::GPUTag>,
                T, core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_CONVERT;
        }
    };

    template<typename T>
    void tff::kernel::ConvertOP<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            ConvertOPBuilder::Params::In, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            ConvertOPBuilder::Params::Out, para_ptr);
        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        kernel::builder::OpParamBuilderBase<ConvertOPBuilder>::CommonParams::Stream, para_ptr);
        auto name = kernel::base::get_param_value<std::string>(
            kernel::builder::OpParamBuilderBase<ConvertOPBuilder>::CommonParams::Name, para_ptr);
        if ( input_tensor == nullptr || input_tensor->get_buffer() == nullptr || output_tensor == nullptr) {
            log::Logger::error("node %s input or output is null", name.c_str());
            return;
        }
        const int n = output_tensor->get_bytes() / sizeof(T);
        fp32_to_fp16<T>(static_cast<T *>(output_tensor->get_buffer()->ptr()),
            static_cast<float *>(input_tensor->get_buffer()->ptr()), n, stream);
    }

    template class tff::kernel::ConvertOP<half, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(ConvertOP, half, core::device::GPUTag);
}
