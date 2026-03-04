//
// Created by nkk on 2025/11/20.
//
#include "device/cuda/cudaInc.h"
#define THREAD_MASK 0xffffffff
template<int block_size, int warp_size>
__global__ void rms_norm_kernel_cuda_impl(
    const int B,
    const int S,
    const int D,
    const float *const src,
    const float *const weight,
    float *const dst) {
    const int batch_index = blockIdx.y;
    const int row_index = blockIdx.x;
    const int thread_id = threadIdx.x;
    if (row_index > B * S) {
        return;
    }
    const float *row_data = src + batch_index * S * D + row_index * D;
    //
    float sub_sum_seq = 0.0f;
#pragma unroll
    for (int d = thread_id; d < D / warp_size; d += block_size) {
        float value = *(row_data + d);
        sub_sum_seq += value * value;
    }
    //
#pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sub_sum_seq += __shfl_xor_sync(THREAD_MASK, sub_sum_seq, offset, warp_size);
    }

}

template<typename T>
void rms_norm_kernel_cpu(
    const int B,
    const int S,
    const int D,
    T *const src,
    T *const weight,
    T *const dst,
    float eps) {
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            float sum_seq = 0.0f;
            int current_token_row_id = (b * S + s) * D;
            for (int d = 0; d < D; d++) {
                auto &token_value = src[current_token_row_id + d];
                sum_seq += token_value * token_value;
            }
            //
            float rms = std::sqrt(sum_seq / static_cast<float>(D) + eps);
            float inv_rms = 1.0f / rms;
            for (int d = 0; d < D; ++d) {
                dst[current_token_row_id + d] = (src[current_token_row_id + d] * inv_rms) * weight[d];
            }
        }
    }
}
