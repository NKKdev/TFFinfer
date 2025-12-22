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
const int WARP_NUM_PER_BLOCK = 8;
template<const int WARP_SIZE, const int BLOCK_SIZE, const int WARP_NUM_PER_BLOCK>
__global__ void quant_q_8_0(const float * __restrict__ src,
    void *dst, const int64_t M, const int64_t N, const int ld, const int dst_stride_cnt) {
    const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = g_thread_id / WARP_SIZE;
    const int lane_id = g_thread_id % WARP_SIZE;

    const int start_block = blockIdx.x * WARP_NUM_PER_BLOCK * WARP_SIZE;
    const int g_index = start_block + warp_id * BLOCK_SIZE + lane_id;

    const int g_dst_index = blockIdx.x * WARP_NUM_PER_BLOCK + warp_id;
    auto *dst_ptr = static_cast<tff::core::quant::Q_8_0 *>(dst);

    float x = 0.0f;
    float max_value = 0.0f;
    if (g_index < N * M) {
        x = src[g_index];
        max_value = fabsf(x);
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
    }

    float d = max_value / 127.0f;
    if (g_dst_index < M * N / BLOCK_SIZE) {
        if (lane_id == 0) {
            dst_ptr[g_dst_index].d = __float2half(d);
        }
        dst_ptr[g_dst_index].qs[lane_id] = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
    }

}
template<const int WARP_SIZE, const int BLOCK_SIZE>
__global__ void quant_q_8_0_2d(const float * __restrict__ src,
    void *dst, const int M, const int N, const int ld, const int dst_stride_cnt) {
    const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = g_thread_id / WARP_SIZE;
    const int lane_id = g_thread_id % WARP_SIZE;

    const int row = blockIdx.y * blockDim.y + warp_id;
    const int col = blockIdx.x * blockDim.x + lane_id;
    const int start_dst_row = row ;
    const int start_dst_col = col / BLOCK_SIZE;
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
    float d = max_value / 127.0f;
    const int index = start_dst_row * dst_stride_cnt + start_dst_col ;
    if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
        if (lane_id == 0) {
            dst_ptr[index].d = __float2half(d);
        }
        dst_ptr[index].qs[lane_id] = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
    }

}
template<const int WARP_SIZE, const int BLOCK_SIZE>
__global__ void quant_q_8_0_2d_reshape(const float * __restrict__ src,
    half *scale_ptr,
    int8_t *quant_ptr,
    const int M, const int N, const int ld, const int dst_stride_cnt) {
    const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = g_thread_id / WARP_SIZE;
    const int lane_id = g_thread_id % WARP_SIZE;

    const int row = blockIdx.y * blockDim.y + warp_id;
    const int col = blockIdx.x * blockDim.x + lane_id;
    const int start_dst_row = row ;
    const int start_dst_col = col / BLOCK_SIZE;


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
    float d = max_value / 127.0f;
    const int index = start_dst_row * dst_stride_cnt + start_dst_col ;
    if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
        if (lane_id == 0) {
            scale_ptr[index] = __float2half(d);
        }
        *(quant_ptr + start_dst_row * ld + col) = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
    }
}

template<const int BLOCK_SIZE>
__global__ void dequant_q_8_0(const int M, const int N, const int ld,
                              const Q_8_0 *src, float *dst,
                              const int dst_stride_cnt) {
    const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = g_thread_id / 32;
    const int lane_id = g_thread_id % 32;

    const int row = blockIdx.y * blockDim.y + warp_id;
    const int col = blockIdx.x * blockDim.x + lane_id;
    const int start_dst_row = row ;
    const int start_dst_col = col / BLOCK_SIZE;


    const int index = start_dst_row * dst_stride_cnt + start_dst_col ;
    if (start_dst_row >= M && start_dst_col >= dst_stride_cnt) {
        return;
    }
    float scale = 0.0f;
    if (lane_id == 0) {
        scale = __half2float(src[index].d);
    }
    scale = __shfl_sync(0xFFFFFFFF, scale, 0);

    if (col < N && row < M) {
        dst[row * ld + col] = static_cast<float>(src[index].qs[lane_id]) * scale;
    }
}
#include <cstdint>

inline uint32_t div_u32_cpu(uint32_t n, uint32_t magic, int shift) {
    // 计算 (n * magic) 的高32位：等价于 (uint64_t)n * magic >> 32
    uint64_t product = (uint64_t)n * magic;
    uint32_t high_part = (uint32_t)(product >> 32);
    return (high_part + n) >> shift;
}
static void dequantize_func(Q_8_0 *blocks, float *out, const int64_t elem_count) {
    const int nb = elem_count / BLOCK_SIZE;
    for (int i = 0; i < nb; ++i) {
        const float scale = __half2float(blocks[i].d);
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            out[i * BLOCK_SIZE + j] = blocks[i].qs[j] * scale;
        }
    }
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

        blocks[i].d = half(scale);

        for (int j = 0; j < BLOCK_SIZE; ++j) {
            const float v = x[j] * inv_scale;
            const int32_t iv = static_cast<int32_t>(std::round(v));
            blocks[i].qs[j] = static_cast<int8_t>(
                std::max(-127, std::min(127, iv))
            );
            // if (i == 156 * 512 / BLOCK_SIZE + 10 && j == 19) {
            //     printf("half d: %f\n", (blocks[i].d));
            //     printf("d: %f\n", scale);
            //     printf("x: %f\n", x);
            //     printf("x / d: %f\n", v);
            //     printf("roundf(x / d): %f\n", std::round(v));
            //     printf("static_cast<int32_t>(roundf(x / d)): %d\n", static_cast<int32_t>(std::round(v)));
            //     printf("static_cast<int8_t>(roundf(x / d)): %d\n", static_cast<int8_t>(static_cast<int32_t>(std::round(v))));
            // }
        }
    }
}
template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}
void quant_q_8_0_1d(const int64_t m, const int64_t n, std::vector<float> &src, std::vector<tff::core::quant::Q_8_0> &dst) {
    dst.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);

    std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
    c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
#ifdef _DEBUG
    quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
#endif
    float *src_gpu = nullptr;
    cudaMalloc(&src_gpu, sizeof(float) * m * n);
    cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
    void *dst_gpu = nullptr;
    cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);

    const int WARP_SIZE = 32;
    dim3 grid((n * m + WARP_NUM_PER_BLOCK * WARP_SIZE - 1) / (WARP_NUM_PER_BLOCK * WARP_SIZE), 1, 1);
    dim3 block(WARP_NUM_PER_BLOCK * WARP_SIZE, 1, 1);
    printf("grid:%d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    quant_q_8_0<32, BLOCK_SIZE, WARP_NUM_PER_BLOCK><<<grid, block>>>(src_gpu, dst_gpu, m, n, n, n / BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(tff::core::quant::Q_8_0),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(dst_gpu);
    dst_gpu = nullptr;
    cudaFree(src_gpu);
    src_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Matrix size: %d x %d\n", m, n);
    printf("Kernel time: %.4f ms\n", milliseconds);
#ifdef _DEBUG
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n / BLOCK_SIZE; j++) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (dst[i * n / BLOCK_SIZE + j].qs[k]);
                //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
                if (delta_qs > 1.0f) {
                    printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i,j, k,delta_qs);
                    return;
                }
            }
            //printf("\n");
        }
        //printf("\n");
    }
#endif
    printf("1d success!!\n");
}
void quant_q_8_0_2d(const int m, const int n, std::vector<float> &src, std::vector<tff::core::quant::Q_8_0> &dst) {

    dst.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);

    std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
    c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
#ifdef _DEBUG
    quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
#endif
    float *src_gpu = nullptr;
    cudaMalloc(&src_gpu, sizeof(float) * m * n);
    cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
    void *dst_gpu = nullptr;
    cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);


    const int WARP_SIZE = 32;
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
    dim3 block(WARP_SIZE, WARP_NUM_PER_BLOCK, 1);
    printf("grid:%d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    quant_q_8_0_2d<32, BLOCK_SIZE><<<grid, block>>>(src_gpu, dst_gpu, m, n, n, n / BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(tff::core::quant::Q_8_0),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(dst_gpu);
    dst_gpu = nullptr;
    cudaFree(src_gpu);
    src_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Matrix size: %d x %d\n", m, n);
    printf("Kernel time: %.4f ms\n", milliseconds);
#ifdef _DEBUG
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n / BLOCK_SIZE; j++) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (dst[i * n / BLOCK_SIZE + j].qs[k]);
                //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
                if (delta_qs > 1.0f) {
                    printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i,j, k,delta_qs);
                    return;
                }
            }
            //printf("\n");
        }
        //printf("\n");
    }
#endif
    printf("2d success!!\n");
}
void dequant_q_8_0_2d(const int m, const int n, std::vector<float> &src, std::vector<tff::core::quant::Q_8_0> &dst) {

    dst.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);

    std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
    c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
#ifdef _DEBUG
    quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
#endif
    float *src_gpu = nullptr;
    cudaMalloc(&src_gpu, sizeof(float) * m * n);
    cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
    void *dst_gpu = nullptr;
    cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);


    const int WARP_SIZE = 32;
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
    dim3 block(WARP_SIZE, WARP_NUM_PER_BLOCK, 1);
    printf("grid:%d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    quant_q_8_0_2d<32, BLOCK_SIZE><<<grid, block>>>(src_gpu, dst_gpu, m, n, n, n / BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(tff::core::quant::Q_8_0),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);


    printf("Matrix size: %d x %d\n", m, n);
    printf("Kernel time: %.4f ms\n", milliseconds);
#ifdef _DEBUG
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n / BLOCK_SIZE; j++) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (dst[i * n / BLOCK_SIZE + j].qs[k]);
                //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
                if (delta_qs > 1.0f) {
                    printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i,j, k,delta_qs);
                    return;
                }
            }
            //printf("\n");
        }
        //printf("\n");
    }
#endif
    printf("2d quant success!!\n");


    std::vector<float> dequant_result;
    dequant_result.resize(m * n);

    float *dequant_gpu = nullptr;
    cudaMalloc(&dequant_gpu, sizeof(float) * m * n);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    dequant_q_8_0<BLOCK_SIZE><<<grid, block>>>(m, n, n, static_cast<Q_8_0 *>(dst_gpu),dequant_gpu, n / BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(dequant_result.data(), dequant_gpu, dequant_result.size() * sizeof(float),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);


    cudaFree(dst_gpu);
    dst_gpu = nullptr;
    cudaFree(src_gpu);
    src_gpu = nullptr;
    cudaFree(dequant_gpu);
    dequant_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

#ifdef _DEBUG
    std::vector<float> dequant_result_cpu;
    dequant_result_cpu.resize(m * n);
    dequantize_func(dst.data(), dequant_result_cpu.data(), m * n);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
           float delta = dequant_result_cpu[i * n + j] - dequant_result[i * n + j];
            if (delta > 0.001f) {
                printf(" dequant error delta: %lf,cpu[%d]: %lf, gpu[%d]: %lf ", delta, i * n + j, dequant_result_cpu[i * n + j],i * n + j,  dequant_result[i * n + j]);
                return;
            }
        }
        //printf("\n");
    }
    printf(("dequant success!!\n"));
#endif

}
void quant_q_8_0_2d_reshape(const int m, const int n, std::vector<float> &src, std::vector<tff::core::quant::Q_8_0> &dst) {

    dst.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
    std::vector<half> scale_result;
    scale_result.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
    std::vector<int8_t> quant_result_cpu;
    quant_result_cpu.resize(m * n);

    std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
    c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
#ifdef _DEBUG
    quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
#endif
    float *src_gpu = nullptr;
    cudaMalloc(&src_gpu, sizeof(float) * m * n);
    cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
    void *dst_gpu = nullptr;
    cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
    auto b_scaled_gpu = reinterpret_cast<half *>(dst_gpu);
    auto b_quant_gpu = reinterpret_cast<int8_t *>(dst_gpu + m * n / tff::core::quant::Q_8_0::BLOCK_SIZE * sizeof(half));

    const int WARP_SIZE = 32;
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
    dim3 block(WARP_SIZE, WARP_NUM_PER_BLOCK, 1);
    printf("grid:%d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    quant_q_8_0_2d_reshape<32, BLOCK_SIZE><<<grid, block>>>(src_gpu, b_scaled_gpu,b_quant_gpu, m, n, n, n / BLOCK_SIZE);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(scale_result.data(), b_scaled_gpu, dst.size() * sizeof(half),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaMemcpy(quant_result_cpu.data(), b_quant_gpu, m * n * sizeof(int8_t),
           cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(dst_gpu);
    dst_gpu = nullptr;
    cudaFree(src_gpu);
    src_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Matrix size: %d x %d\n", m, n);
    printf("Kernel time: %.4f ms\n", milliseconds);


#ifdef _DEBUG
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n / BLOCK_SIZE; j++) {
            auto *current_block = &quant_result_cpu[i * n + j * BLOCK_SIZE];
            auto current_scale = scale_result[i * n / BLOCK_SIZE + j];
            for (int k = 0; k < BLOCK_SIZE; k++) {
                float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (current_block[k]);
                //float delta_d = __half2float(c_mat_quant_result_cpu[i * n + j].d) - __half2float(current_scale);
                if (delta_qs > 1) {
                    printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i,j, k,delta_qs);
                    return;
                }
            }
            //printf("\n");
        }
        //printf("\n");
    }
#endif
    printf("2d success!!\n");
}
int main345(int argc , char *argv[]) {
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, 0);
    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-127, 127);
    int m = 64;
    int n = 64;

    std::vector<float> c_mat;
    c_mat.resize(m * n);
    PopulateVector<float>(c_mat, mt, dist);
    std::vector<tff::core::quant::Q_8_0> dst;
    //quant_q_8_0_1d(m,n, c_mat, dst);
    //quant_q_8_0_2d(m,n, c_mat, dst);
    //quant_q_8_0_2d_reshape(m,n, c_mat, dst);
    dequant_q_8_0_2d(m, n, c_mat, dst);
    return 0;
}