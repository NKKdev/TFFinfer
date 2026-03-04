//
// Created by nkk on 2026/1/21.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#define _PTX
#define _ASYNC
#define _DOUBER_BUFFER

namespace tff::kernel {
    template<typename T, const int BLOCK_DIM_M, const int BLOCK_DIM_N>
    __global__ void autoregressive_mask(const int M, const int N, T *__restrict__ mask) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        const int thread_x = thread_id % 32;
        const int warp_id = thread_id / 32;
        const int64_t block_x = blockIdx.x * BLOCK_DIM_N;
        const int64_t block_y = blockIdx.y * BLOCK_DIM_M;
        const int VEC_DIM_M = BLOCK_DIM_M / 8;
        const int VEC_DIM_N = BLOCK_DIM_N / 32;

#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            const int64_t g_row = (block_y + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M);
            if (g_row >= M) {
                continue;
            }
#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                const int64_t g_col = (block_x + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N);
                if (g_col >= N) {
                    continue;
                }
                const int64_t g_index = g_row * N + g_col;

                if (std::is_same_v<T, half>) {
                    T neg_inf = __float2half(-INFINITY); // 将 -inf 转为 half
                    if (g_row >= g_col) {
                        mask[g_index] = __float2half(0.0f);
                    } else {
                        mask[g_index] = neg_inf;
                    }
                } else if (std::is_same_v<T, float>) {
                    mask[g_index] = -INFINITY;
                    if (g_col < g_row) {
                        mask[g_index] = (0.0f);
                    }
                }
            }
        }
    }

    //
    template<typename T>
    class MaskOP<T,
                core::device::GPUTag> : public base::OPCreatorBase<MaskOP<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_ATTN_MASK;
        }
    };

    template<typename T>
    void get_autoregressive_mask(const core::graph::TFFMaskType &mask_type,
                                 std::shared_ptr<core::memory::Tensor> &mask,
                                 std::shared_ptr<core::device::DeviceStream> &stream) {
        switch (mask_type) {
            case core::graph::TFFMaskType::TFF_MASK_TYPE_CAUSAL: {
                const int M = mask->get_shape()[1];
                const int N = mask->get_shape()[0];
                constexpr int BLOCK_DIM_M = 64;
                constexpr int BLOCK_DIM_N = 64;
                dim3 grid((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, (M + BLOCK_DIM_M - 1) / BLOCK_DIM_M);
                dim3 block(32, 8);

                autoregressive_mask<T, BLOCK_DIM_M, BLOCK_DIM_N><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                    get_native_stream())>>>(
                    M, N, static_cast<T *>(mask->get_buffer()->ptr()));
                break;
            }
            default:
                break;
        }
    }

    //
    template<typename T>
    void tff::kernel::MaskOP<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto mask_type = static_cast<tff::core::graph::TFFMaskType>(kernel::base::get_param_value<const int>(
            MaskOPBuilder::Params::MaskType, para_ptr));
        auto mask = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(MaskOPBuilder::Params::Out, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            kernel::builder::OpParamBuilderBase<MemOptOPBuilder>::CommonParams::Stream, para_ptr);
        if (mask == nullptr) {
            return;
        }
        get_autoregressive_mask<T>(mask_type, mask, stream);
    }

    template class tff::kernel::MaskOP<half, core::device::GPUTag>;
    template class tff::kernel::MaskOP<float, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(MaskOP, half, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(MaskOP, float, core::device::GPUTag);
}
