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
#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                const int64_t g_col = (block_x + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N);
                const int64_t g_index = g_row * N + g_col;
                if (g_index > int64_t(M * N)) {
                    return;
                }
                if (std::is_same_v<T, half>) {
                    T neg_inf = __float2half(-INFINITY); // 将 -inf 转为 half
                    mask[g_index] = neg_inf;
                    if (g_col < g_row) {
                        mask[g_index] = __float2half(0.0f);
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
                dim3 grid((N + 256 - 1) / 256, (M + 256 - 1) / 256);
                dim3 block(32, 8);

                autoregressive_mask<T, BLOCK_DIM_M, BLOCK_DIM_N><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>>(
                    M, N, static_cast<T *>(mask->get_buffer()->ptr()));
                break;
            }
            default:
                break;
        }
    }

    //
    template<typename T>
    void tff::kernel::MaskOP<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto mask_type = static_cast<tff::core::graph::TFFMaskType>(kernel::base::get_param_value<const int>(0, para_ptr));
        auto mask = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(1, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        para_ptr->get_param_count() - 1, para_ptr);

        get_autoregressive_mask<T>(mask_type, mask, stream);
    }

    template class tff::kernel::MaskOP<half>;
    template class tff::kernel::MaskOP<float>;
    REGISTER_OP_OBJECT(MaskOP, half);

    REGISTER_OP_OBJECT(MaskOP, float);
}
