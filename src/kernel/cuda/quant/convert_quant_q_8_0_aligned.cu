//
// Created by nkk on 2026/1/28.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {
    template<typename T, const int WARP_SIZE, const int BLOCK_SIZE>
    __global__ __forceinline__ void quant_aligned_q_8_0(const Q8_0 *__restrict__ src,
                                                        T *dst, const int M,
                                                        const int dst_stride_cnt) {
        const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = g_thread_id / WARP_SIZE;
        const int lane_id = g_thread_id % WARP_SIZE;

        const int row = blockIdx.y * blockDim.y + warp_id;
        const int col = blockIdx.x * blockDim.x + lane_id;
        const int start_dst_row = row;
        const int start_dst_col = col / BLOCK_SIZE;

        const int index = start_dst_row * dst_stride_cnt + start_dst_col;
        if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
            auto src_val = &src[index];
            if (lane_id == 0) {
                dst[index].d = __half2float(src_val->d);
            }
            dst[index].qs[lane_id] = src_val->qs[lane_id];
        }
    }

    template<typename T>
    void quant_aligned(const int M, const int N,
                       std::shared_ptr<tff::core::memory::Tensor> &src,
                       std::shared_ptr<tff::core::memory::Tensor> &dst,
                       std::shared_ptr<core::device::DeviceStream> &stream) {
        if constexpr (std::is_same_v<T, Q8_0_ALIGNED>) {
            constexpr int BLOCK_SIZE = tff::core::quant::Q_8_0::BLOCK_SIZE;
            constexpr int VEC_M_DIM = 8;
            constexpr int WARP_NUM_PER_BLOCK = 8;

            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
            dim3 block(32, WARP_NUM_PER_BLOCK, 1);
            quant_aligned_q_8_0<T, 32, BLOCK_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                        get_native_stream())>>>
                    (static_cast<Q8_0 *>(src->get_buffer()->ptr()),
                     static_cast<T *>(dst->get_buffer()->ptr()), M, N / BLOCK_SIZE);
        }
    }
    template<typename T>
    class QuantAligned<T, core::device::GPUTag> : public base::OPCreatorBase<QuantAligned<T, core::device::GPUTag>, T, core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_QUANTIZE_ALIGNED;
        }
    };
    template<typename T>
    void tff::kernel::QuantAligned<T, core::device::GPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            QuantAlignedBuilder::Params::In, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            QuantAlignedBuilder::Params::Out, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
               kernel::builder::OpParamBuilderBase<QuantAlignedBuilder>::CommonParams::Stream, para_ptr);

        if (input_tensor == nullptr || input_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("input_tensor buffer is nullptr!");
            return;
        }
        if (output_tensor == nullptr || output_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("output_tensor buffer is nullptr!");
            return;
        }

        const int M = input_tensor->get_shape()[1];
        const int N = input_tensor->get_shape()[0];
        quant_aligned<T>(M, N, input_tensor, output_tensor, stream);

    }


    template class tff::kernel::QuantAligned<Q8_0_ALIGNED, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(QuantAligned, Q8_0_ALIGNED, core::device::GPUTag);
}
