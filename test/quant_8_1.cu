//
// Created by nkk on 2025/11/29.
//

#include <vector>
#include <random>
#include "cublas_v2.h"
#include "mma.h"
#include <cstdint>
#include <cstring>
#include "core/quant/BaseDefine.h"
#include "device/cuda/cudaInc.h"
#include "include/kernel_util.h"
using namespace tff::core::quant;
const int MAX_RANGE_VALUE = 127;
const int BLOCK_SIZE = 32;
const int VEC_M_DIM = 8;

template<const int WARP_SIZE, const int BLOCK_SIZE, const int VEC_M_DIM, const int MAX_RANGE_VALUE>
__global__ void quant_q_8_0(const float * __restrict__ src,
    void *dst, const int M, const int N, const int ld, const int dst_stride_cnt,
    const int magic127, const int shift127,
    const int magic32, const int shift32) {
    const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = g_thread_id / WARP_SIZE;
    const int lane_id = g_thread_id % WARP_SIZE;

    const int row = blockIdx.y * blockDim.y + warp_id;
    const int col = blockIdx.x * blockDim.x + lane_id;
    const int start_dst_row = row ;
    const int start_dst_col = col / 32;//* tff::kernel::div_u32(BLOCK_SIZE, magic32, shift32);
    auto *dst_ptr = static_cast<tff::core::quant::Q_8_0 *>(dst);

    float x = 0.0f;
    float max_value = 0.0f;
    if (col < N && row < M) {
        x = src[row * ld + col];
        max_value = fabsf(x);
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
    }
    //float d = max_value * tff::kernel::div_u32(MAX_RANGE_VALUE, magic127, shift127);
    float d = max_value / 127.0f;
    const int index = start_dst_row * dst_stride_cnt + start_dst_col ;
    dst_ptr[index].d = __float2half(d);
    dst_ptr[index].qs[lane_id] = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
}
static void quantize_func(const float *src, Q_8_0 *blocks, const int64_t elem_count) {
    const int nb = static_cast<int>(elem_count / BLOCK_SIZE);
    for (int i = 0; i < nb; ++i) {
        const float *x = src + i * BLOCK_SIZE;

        float max_abs = 0.0f;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            max_abs = std::max(max_abs, std::abs(x[j]));
        }
        if (max_abs == 0.0f) {
            blocks[i].d = tff::utils::fp32_to_fp16(0.0f);
            for (int j = 0; j < BLOCK_SIZE; ++j) {
                blocks[i].qs[j] = 0;
            }
            continue;
        }
        const float scale = max_abs / 127.0f;
        const float inv_scale = 1.0f / scale;

        blocks[i].d = tff::utils::fp32_to_fp16(scale);

        for (int j = 0; j < BLOCK_SIZE; ++j) {
            const float v = x[j] * inv_scale;
            const int32_t iv = static_cast<int32_t>(std::round(v));
            blocks[i].qs[j] = static_cast<int8_t>(
                std::max(-127, std::min(127, iv))
            );
        }
    }
}
template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}
int main(int argc , char *argv[]) {
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, 0);
    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-127, 127);
    int m = 8;
    int n = 32;

    std::vector<float> c_mat;
    c_mat.resize(m * n);
    std::vector<tff::core::quant::Q_8_0> c_mat_quant_result;
    c_mat_quant_result.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);

    std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
    c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);

    PopulateVector<float>(c_mat, mt, dist);


    quantize_func(c_mat.data(), c_mat_quant_result_cpu.data(), m * n);

    float *src_gpu = nullptr;
    cudaMalloc(&src_gpu, sizeof(float) * m * n);
    cudaMemcpy(src_gpu, c_mat.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
    void *dst_gpu = nullptr;
    cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);




    auto [magic127, shift127] = tff::utils::gen_magic_u32(MAX_RANGE_VALUE);
    auto [magic, shift] = tff::utils::gen_magic_u32(BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + VEC_M_DIM - 1) / VEC_M_DIM, 1);
    dim3 block(BLOCK_SIZE, VEC_M_DIM, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    quant_q_8_0<32, BLOCK_SIZE, VEC_M_DIM, MAX_RANGE_VALUE><<<grid, block>>>(src_gpu, dst_gpu, m, n, n, n / BLOCK_SIZE,
        magic127, shift127, magic, shift);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_quant_result.data(), dst_gpu, c_mat_quant_result.size() * sizeof(tff::core::quant::Q_8_0),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(dst_gpu);
    dst_gpu = nullptr;
    cudaFree(src_gpu);
    src_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Matrix size: %d x %d x %d\n", m, n);
    printf("Kernel time: %.4f ms\n", milliseconds);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n / BLOCK_SIZE; j++) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                bool delta_qs = (char)(c_mat_quant_result_cpu[i * n + j].qs[k]) == (char)(c_mat_quant_result[i * n + j].qs[k]);
                float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
                if (!delta_qs) {
                    printf("m: %d, n: %d, block_index: %d,error qs diff: %d, d diff: %f \n", i,j, k,delta_qs, delta_d);
                }
            }
            printf("\n");
        }
        printf("\n");
    }
    return 0;
}