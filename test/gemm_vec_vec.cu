//
// Created by nkk on 2025/12/5.
//

#include <vector>
#include <random>
#include "cublas_v2.h"
#include "mma.h"
#include <cstdint>
#include <cstring>

#include "../cmake-build-debug/_deps/fmt-src/include/fmt/base.h"

// #include "cutlass/gemm/device/gemm.h"
// #include "cutlass/gemm/device/gemm_universal_adapter.h"
// #include "cutlass/numeric_types.h"
// #include "cutlass/layout/matrix.h"
// #include <cuda_runtime.h>
// #include <iostream>

using T = float;
#if 1
constexpr int BYTES_PER_LOAD = 16; // 128-bit
constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD / sizeof(T);
constexpr int VEC_DIM_LOAD = 2 * ELEMENTS_PER_LOAD;
constexpr int WARP_SIZE = 32;
constexpr int THREAD_BLOCK_SIZE = 256;
constexpr int BLOCK_DIM_K = 16;
constexpr int VEC_DIM_N = 8;
constexpr int VEC_DIM_K = 1;
constexpr int VEC_DIM_M = 8;
constexpr int BLOCK_DIM_M = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_LOAD;
constexpr int BLOCK_DIM_N = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_LOAD;
constexpr int PAD_SIZE = 16; //BLOCK_DIM_K;
#else
constexpr int BYTES_PER_LOAD = 16; // 128-bit
constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD / sizeof(T);
constexpr int VEC_DIM_LOAD = ELEMENTS_PER_LOAD;
constexpr int WARP_SIZE = 32;
constexpr int THREAD_BLOCK_SIZE = 256;
constexpr int BLOCK_DIM_K = 32;
constexpr int VEC_DIM_N = 8;
constexpr int VEC_DIM_K = 4;
constexpr int VEC_DIM_M = 8;
constexpr int BLOCK_DIM_M = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_LOAD;
constexpr int BLOCK_DIM_N = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_LOAD;
constexpr int PAD_SIZE = 16; //BLOCK_DIM_K;
#endif

template<typename T>
__device__ __forceinline__ void load_vec(const T *addr, T *out, int count);

template<>
__device__ __forceinline__ void load_vec<float>(const float *addr, float *out, int count) {
    if (count >= 4 && reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
        float4 v = *reinterpret_cast<const float4 *>(addr);
        out[0] = v.x;
        out[1] = v.y;
        out[2] = v.z;
        out[3] = v.w;
    } else {
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
    }
}

template<>
__device__ __forceinline__ void load_vec<half>(const half *addr, half *out, int count) {
    if (count >= 8 && reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
        uint4 v = *reinterpret_cast<const uint4 *>(addr);
        const half *h = reinterpret_cast<const half *>(&v);
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = h[i];
    } else {
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
    }
}

template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int
    PAD_SIZE>
__device__ void load_tile_n_vec(const int ld, const int dim,
                                const int thread_x, const int thread_y,
                                const int start_m,
                                const int k,
                                const T *__restrict__ global_mem,
                                T *sm) {
#pragma unroll
    for (int j = 0; j < VEC_DIM_LD / ELEMENTS_PER_LOAD; ++j) {
        const int dim0_base = start_m + thread_x * ELEMENTS_PER_LOAD + j * (BLOCK_DIM_LD / VEC_DIM_LD) *
                              ELEMENTS_PER_LOAD;
        for (int kk = 0; kk < VEC_DIM_K; ++kk) {
            const int dim1 = k + thread_y + kk * (BLOCK_DIM_K / VEC_DIM_K);
            T val[ELEMENTS_PER_LOAD] = {0};
            if (dim1 < dim) {
                const int actual_load = min(ELEMENTS_PER_LOAD, ld - dim0_base);
                if (actual_load > 0) {
                    load_vec<T>(&global_mem[dim1 * ld + dim0_base], val, actual_load);
                }
            }
            int sm_row = thread_y + kk * (BLOCK_DIM_K / VEC_DIM_K);
            int sm_col_base = thread_x * ELEMENTS_PER_LOAD + j * (BLOCK_DIM_LD / VEC_DIM_LD) * ELEMENTS_PER_LOAD;
#pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LOAD; ++i) {
                sm[sm_row * (BLOCK_DIM_LD + PAD_SIZE) + sm_col_base + i] = val[i];
            }
        }
    }
}

template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int
    PAD_SIZE>
__device__ void load_tile_n(const int ld, const int dim,
                            const int thread_x, const int thread_y,
                            const int start_m,
                            const int k,
                            const T *__restrict__ global_mem,
                            T *sm) {
#pragma unroll
    for (int j = 0; j < VEC_DIM_LD; j++) {
        int dim0 = start_m + thread_x + j * BLOCK_DIM_LD / VEC_DIM_LD;
        for (int kk = 0; kk < VEC_DIM_K; kk++) {
            int dim1 = k + thread_y + kk * BLOCK_DIM_K / VEC_DIM_K;
            T val = 0.0f;
            if (dim1 < dim && dim0 < ld) {
                val = __ldg(&global_mem[dim1 * ld + dim0]);
            }
            sm[(thread_y + kk * BLOCK_DIM_K / VEC_DIM_K) * (BLOCK_DIM_LD + PAD_SIZE) + thread_x + j * BLOCK_DIM_LD /
               VEC_DIM_LD] = val;
        }
    }
}

template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int
    PAD_SIZE>
__device__ void load_tile_t(const int ld, const int dim,
                            const int thread_x, const int thread_y,
                            const int start_block,
                            const int k,
                            const T *__restrict__ global_mem,
                            T *sm) {
#pragma unroll
    for (int j = 0; j < VEC_DIM_LD; j++) {
        int dim0 = start_block + thread_x + j * BLOCK_DIM_LD / VEC_DIM_LD;
        for (int kk = 0; kk < VEC_DIM_K; kk++) {
            int dim1 = k + thread_y + kk * BLOCK_DIM_K / VEC_DIM_K;
            T val = {0.0f};
            if (dim0 < dim && dim1 < ld) {
                val = __ldg(&global_mem[dim1 + dim0 * ld]);
            }
            sm[(thread_y + kk * BLOCK_DIM_K / VEC_DIM_K) * (BLOCK_DIM_LD + PAD_SIZE) + thread_x
               + j * BLOCK_DIM_LD / VEC_DIM_LD] = val;
        }
    }
}
template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int
    PAD_SIZE>
__device__ void load_tile_t_vec(const int ld, const int dim,
                            const int thread_x, const int thread_y,
                            const int start_block,
                            const int k,
                            const T *__restrict__ global_mem,
                            T *sm) {
#pragma unroll
    for (int j = 0; j < VEC_DIM_LD; j++) {
        int dim0_base = start_block + thread_x + j * BLOCK_DIM_LD / VEC_DIM_LD;
        for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; kk++) {
            int dim1 = k + thread_y * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD;
            T val[ELEMENTS_PER_LOAD] = {0};
            if (dim0_base < dim) {
                const int actual_load = min(ELEMENTS_PER_LOAD, ld - dim1);
                if (actual_load > 0) {
                    load_vec<T>(&global_mem[dim0_base * ld + dim1], val, actual_load);
                }
            }
#pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LOAD; ++i) {
                sm[(thread_y * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD + i) * (BLOCK_DIM_LD + PAD_SIZE) + thread_x
               + j * BLOCK_DIM_LD / VEC_DIM_LD] = val[i];
            }

        }
    }
}
template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__device__ void compute_tile(const int thread_x, const int thread_y,
                             T *a_sm, T *b_sm,
                             float *c_reg) {
    // float a_reg[VEC_DIM_M];
    // float b_reg[VEC_DIM_N];
#pragma unroll
    for (int kk = 0; kk < BLOCK_DIM_K / VEC_DIM_K; kk++) {
        for (int kkk = 0; kkk < VEC_DIM_K; kkk++) {
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                float a_reg = 0.0f;
                if constexpr (std::is_same_v<T, half>) {
                    a_reg = __half2float(
                        a_sm[(kk * VEC_DIM_K + kkk) * (BLOCK_DIM_M + PAD_SIZE) + thread_x + mm * BLOCK_DIM_M /
                             VEC_DIM_M]);
                } else if constexpr (std::is_same_v<T, float>) {
                    a_reg = (a_sm[(kk * VEC_DIM_K + kkk) * (BLOCK_DIM_M + PAD_SIZE) + thread_x + mm * BLOCK_DIM_M /
                                      VEC_DIM_M]);
                }
#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N; nn++) {
                    float b_reg = 0.0f;
                    if constexpr (std::is_same_v<T, half>) {
                        b_reg = __half2float(
                            b_sm[(kk * VEC_DIM_K + kkk) * (BLOCK_DIM_N + PAD_SIZE) + thread_y + nn * BLOCK_DIM_N /
                                 VEC_DIM_N]);
                    } else if constexpr (std::is_same_v<T, float>) {
                        b_reg = (b_sm[(kk * VEC_DIM_K + kkk) * (BLOCK_DIM_N + PAD_SIZE) + thread_y + nn *
                                          BLOCK_DIM_N / VEC_DIM_N]);
                    }

                    c_reg[nn * VEC_DIM_M + mm] += a_reg * b_reg;
                }
            }
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__device__ void compute_tile_vec(const int thread_x, const int thread_y,
                                 T *a_sm, T *b_sm,
                                 float *c_reg) {
    const half2 *a_sm_half = reinterpret_cast<half2 *>(a_sm);
    const half2 *b_sm_half = reinterpret_cast<half2 *>(b_sm);
    float2 a_reg[VEC_DIM_M / 2];
    float2 b_reg[VEC_DIM_N / 2];
#pragma unroll
    for (int kk = 0; kk < BLOCK_DIM_K / VEC_DIM_K; kk++) {
        for (int kkk = 0; kkk < VEC_DIM_K; kkk++) {
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M / 2; mm++) {
                a_reg[mm] = __half22float2(
                    a_sm_half[(kk * VEC_DIM_K + kkk) * (BLOCK_DIM_M + PAD_SIZE) / 2 + thread_x + mm * 8]);
#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N / 2; nn++) {
                    b_reg[nn] = __half22float2(
                        b_sm_half[(kk * VEC_DIM_K + kkk) * (BLOCK_DIM_N + PAD_SIZE) / 2 + thread_y + nn * 8]);

                    c_reg[nn * 2 * VEC_DIM_M + mm * 2 + 0] += a_reg[mm].x * b_reg[nn].x;
                    c_reg[nn * 2 * VEC_DIM_M + mm * 2 + 1] += a_reg[mm].y * b_reg[nn].x;
                    c_reg[(nn * 2 + 1) * VEC_DIM_M + mm * 2] += a_reg[mm].x * b_reg[nn].y;
                    c_reg[(nn * 2 + 1) * VEC_DIM_M + mm * 2 + 1] += a_reg[mm].y * b_reg[nn].y;
                    if (kk == 0 && mm == 1 && nn == 1 && thread_x == 0 && thread_y == 0) {
                        printf(
                            "thread_x: %d, thread_y: %d,a_reg[%d] x: %lf,a_reg[%d] y: %lf, b_reg[%d] x: %lf,b_reg[%d] y: %lf \n",
                            thread_x, thread_y, mm, a_reg[mm].x, mm, a_reg[mm].y, nn, b_reg[nn].x, nn, b_reg[nn].y);
                        printf("c_reg[%d]: %lf \n", nn * 2 * VEC_DIM_M + mm * 2 + 0,
                               c_reg[nn * 2 * VEC_DIM_M + mm * 2 + 0]);
                        printf("c_reg[%d]: %lf \n", nn * 2 * VEC_DIM_M + mm * 2 + 1,
                               c_reg[nn * 2 * VEC_DIM_M + mm * 2 + 1]);
                        printf("c_reg[%d]: %lf \n", (nn * 2 + 1) * VEC_DIM_M + mm * 2,
                               c_reg[(nn * 2 + 1) * VEC_DIM_M + mm * 2]);
                        printf("c_reg[%d]: %lf \n", (nn * 2 + 1) * VEC_DIM_M + mm * 2 + 1,
                               c_reg[(nn * 2 + 1) * VEC_DIM_M + mm * 2 + 1]);
                    }
                }
            }
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int BLOCK_DIM_M, const int BLOCK_DIM_N>
__device__ void store_tile(const int a_ld, const int b_ld, const int c_ld,
                           const int thread_x, const int thread_y,
                           const int start_m, const int start_n,
                           T *__restrict__ c,
                           float *c_reg) {
#pragma unroll
    for (int mm = 0; mm < VEC_DIM_M; mm++) {
        int m_idx = start_m + thread_x + mm * BLOCK_DIM_M / VEC_DIM_M;
        if (m_idx >= a_ld) continue;
#pragma unroll
        for (int nn = 0; nn < VEC_DIM_N; nn++) {
            int n_idx = start_n + thread_y + nn * BLOCK_DIM_N / VEC_DIM_N;
            if (n_idx >= b_ld) continue;
            c[n_idx * c_ld + m_idx] += (c_reg[nn * VEC_DIM_M + mm]);
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int BLOCK_DIM_M, const int BLOCK_DIM_N>
__device__ void store_tile_vec(const int a_ld, const int b_ld, const int c_ld,
                               const int thread_x, const int thread_y,
                               const int start_m, const int start_n,
                               T *__restrict__ c,
                               float *c_reg) {
#pragma unroll
    for (int mm = 0; mm < VEC_DIM_M; mm++) {
        int m_idx = start_m + thread_x * 2 + mm * BLOCK_DIM_M / VEC_DIM_M;
        if (m_idx >= a_ld) continue;
#pragma unroll
        for (int nn = 0; nn < VEC_DIM_N; nn++) {
            int n_idx = start_n + thread_y * 2 + nn * BLOCK_DIM_N / VEC_DIM_N;
            if (n_idx >= b_ld) continue;
            if constexpr (std::is_same_v<T, half>) {
                c[n_idx * c_ld + m_idx + 0] += __float2half(c_reg[nn * VEC_DIM_M + mm + 0]);
                if ((m_idx + 1) < a_ld) {
                    c[n_idx * c_ld + m_idx + 1] += __float2half(c_reg[nn * VEC_DIM_M + mm + 1]);
                }
                if ((n_idx + 1) < b_ld) {
                    c[(n_idx + 1) * c_ld + m_idx + 0] += __float2half(c_reg[(nn + 1) * VEC_DIM_M + mm + 0]);
                }
                if ((m_idx + 1) < a_ld && (n_idx + 1) < b_ld) {
                    c[(n_idx + 1) * c_ld + m_idx + 1] += __float2half(c_reg[(nn + 1) * VEC_DIM_M + mm + 1]);
                }
            } else if constexpr (std::is_same_v<T, float>) {
                c[n_idx * c_ld + m_idx] += (c_reg[nn * VEC_DIM_M + mm]);
            }
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_nn_pipeline_double_buffer(
    int M, int N, int K,
    int a_ld, int b_ld, int c_ld,
    const T *__restrict__ a,
    const T *__restrict__ b,
    T *__restrict__ c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int thread_x = thread_id % thread_block_n;
    const int thread_y = thread_id / thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M; // 128 * blockIdx.x
    const int start_n = block_y * BLOCK_DIM_N;


    __shared__ T a_sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[2][BLOCK_DIM_K][BLOCK_DIM_N + PAD_SIZE];

    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

    int flip_flag = 0;
    int k = 0;
    load_tile_n_vec<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                             start_m, k, a, &a_sm[flip_flag][0][0]);
    load_tile_t<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                             start_n, k, b, &b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = BLOCK_DIM_K; k <= K; k += BLOCK_DIM_K) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_n_vec<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, thread_x, thread_y,
                start_m, k, a, &a_sm[!flip_flag][0][0]);
            load_tile_t<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, thread_x, thread_y,
                start_n, k, b, &b_sm[!flip_flag][0][0]);
        }

        compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % BLOCK_DIM_K;
        if (remain_k != 0) {
            k = (K / BLOCK_DIM_K) * BLOCK_DIM_K;

            load_tile_n_vec<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, thread_x, thread_y,
                start_m, k, a, &a_sm[flip_flag][0][0]);
            load_tile_t<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, thread_x, thread_y,
                start_n, k, b, &b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
        }
    }
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, thread_x, thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_nn_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                              T *a, T *b, T *c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int ld_thread_block_n = BLOCK_DIM_N / VEC_DIM_LOAD;
    const int ld_thread_x = thread_id % ld_thread_block_n;
    const int ld_thread_y = thread_id / ld_thread_block_n;

    const int cm_thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int cm_thread_x = thread_id % cm_thread_block_n;
    const int cm_thread_y = thread_id / cm_thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M;
    const int start_n = block_y * BLOCK_DIM_N;

    __shared__ T a_sm[BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[BLOCK_DIM_K][BLOCK_DIM_N + PAD_SIZE];
    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
    for (int k = 0; k < K; k += BLOCK_DIM_K) {
        load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, ld_thread_x, ld_thread_y,
            start_m, k, a, &a_sm[0][0]);
        load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, ld_thread_x, ld_thread_y,
            start_n, k, b, &b_sm[0][0]);
        __syncthreads();
        compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            cm_thread_x, cm_thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
        __syncthreads();
    }

    __syncthreads();
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_tt_pipeline_double_buffer(
    int M, int N, int K,
    int a_ld, int b_ld, int c_ld,
    const T *__restrict__ a,
    const T *__restrict__ b,
    T *__restrict__ c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int thread_x = thread_id % thread_block_n;
    const int thread_y = thread_id / thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M; // 128 * blockIdx.x
    const int start_n = block_y * BLOCK_DIM_N;


    __shared__ T a_sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];

    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

    int flip_flag = 0;
    int k = 0;
    load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                             start_m, k, a, &a_sm[flip_flag][0][0]);
    load_tile_n_vec<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                             start_n, k, b, &b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = BLOCK_DIM_K; k <= K; k += BLOCK_DIM_K) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
                start_m, k, a, &a_sm[!flip_flag][0][0]);
            load_tile_n_vec<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, thread_x, thread_y,
                start_n, k, b, &b_sm[!flip_flag][0][0]);
        }

        compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % BLOCK_DIM_K;
        if (remain_k != 0) {
            k = (K / BLOCK_DIM_K) * BLOCK_DIM_K;
            load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
                start_m, k, a, &a_sm[flip_flag][0][0]);
            load_tile_n_vec<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, thread_x, thread_y,
                start_n, k, b, &b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
        }
    }
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, thread_x, thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_tt_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                              T *a, T *b, T *c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int thread_x = thread_id % thread_block_n;
    const int thread_y = thread_id / thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M;
    const int start_n = block_y * BLOCK_DIM_N;

    __shared__ T a_sm[BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[BLOCK_DIM_K][BLOCK_DIM_N + PAD_SIZE];
    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
    for (int k = 0; k < K; k += BLOCK_DIM_K) {
        load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
            start_m, k, a, &a_sm[0][0]);
        load_tile_n_vec<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, thread_x, thread_y,
            start_n, k, b, &b_sm[0][0]);
        __syncthreads();
        compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            thread_x, thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
        __syncthreads();
    }

    __syncthreads();
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, thread_x, thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_tn_pipeline_double_buffer(
    int M, int N, int K,
    int a_ld, int b_ld, int c_ld,
    const T *__restrict__ a,
    const T *__restrict__ b,
    T *__restrict__ c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int thread_x = thread_id % thread_block_n;
    const int thread_y = thread_id / thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M; // 128 * blockIdx.x
    const int start_n = block_y * BLOCK_DIM_N;


    __shared__ T a_sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[2][BLOCK_DIM_K][BLOCK_DIM_N + PAD_SIZE];

    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

    int flip_flag = 0;
    int k = 0;
    load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                             start_m, k, a, &a_sm[flip_flag][0][0]);
    load_tile_t<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                             start_n, k, b, &b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = BLOCK_DIM_K; k <= K; k += BLOCK_DIM_K) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
                start_m, k, a, &a_sm[!flip_flag][0][0]);
            load_tile_t<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, thread_x, thread_y,
                start_n, k, b, &b_sm[!flip_flag][0][0]);
        }

        compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % BLOCK_DIM_K;
        if (remain_k != 0) {
            k = (K / BLOCK_DIM_K) * BLOCK_DIM_K;
            load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
                start_m, k, a, &a_sm[flip_flag][0][0]);
            load_tile_t<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, thread_x, thread_y,
                start_n, k, b, &b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
        }
    }
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, thread_x, thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_tn_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                              T *a, T *b, T *c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int thread_x = thread_id % thread_block_n;
    const int thread_y = thread_id / thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M;
    const int start_n = block_y * BLOCK_DIM_N;

    __shared__ T a_sm[BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[BLOCK_DIM_K][BLOCK_DIM_N + PAD_SIZE];
    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
    for (int k = 0; k < K; k += BLOCK_DIM_K) {
        load_tile_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, thread_x, thread_y,
            start_m, k, a, &a_sm[0][0]);
        load_tile_t<T, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, thread_x, thread_y,
            start_n, k, b, &b_sm[0][0]);
        __syncthreads();
        compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            thread_x, thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
        __syncthreads();
    }

    __syncthreads();
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, thread_x, thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_nt_pipeline_double_buffer(
    int M, int N, int K,
    int a_ld, int b_ld, int c_ld,
    const T *__restrict__ a,
    const T *__restrict__ b,
    T *__restrict__ c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int ld_thread_block_n = BLOCK_DIM_N / VEC_DIM_LOAD;
    const int ld_thread_x = thread_id % ld_thread_block_n;
    const int ld_thread_y = thread_id / ld_thread_block_n;

    const int cm_thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int cm_thread_x = thread_id % cm_thread_block_n;
    const int cm_thread_y = thread_id / cm_thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M; // 128 * blockIdx.x
    const int start_n = block_y * BLOCK_DIM_N;


    __shared__ T a_sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[2][BLOCK_DIM_K][BLOCK_DIM_N + PAD_SIZE];

    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

    int flip_flag = 0;
    int k = 0;
    load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, ld_thread_x, ld_thread_y,
        start_m, k, a, &a_sm[flip_flag][0][0]);
    load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, ld_thread_x, ld_thread_y,
        start_n, k, b, &b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = BLOCK_DIM_K; k <= K; k += BLOCK_DIM_K) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, ld_thread_x, ld_thread_y,
                start_m, k, a, &a_sm[!flip_flag][0][0]);
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, ld_thread_x, ld_thread_y,
                start_n, k, b, &b_sm[!flip_flag][0][0]);
        }

        compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            cm_thread_x, cm_thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % BLOCK_DIM_K;
        if (remain_k != 0) {
            k = (K / BLOCK_DIM_K) * BLOCK_DIM_K;
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, ld_thread_x, ld_thread_y,
                start_m, k, a, &a_sm[flip_flag][0][0]);
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, ld_thread_x, ld_thread_y,
                start_n, k, b, &b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                cm_thread_x, cm_thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
        }
    }
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void sgemm_nt_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                              T *a, T *b, T *c) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int ld_thread_block_n = BLOCK_DIM_N / VEC_DIM_LOAD;
    const int ld_thread_x = thread_id % ld_thread_block_n;
    const int ld_thread_y = thread_id / ld_thread_block_n;

    const int cm_thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int cm_thread_x = thread_id % cm_thread_block_n;
    const int cm_thread_y = thread_id / cm_thread_block_n;

    const int block_x = blockIdx.x;
    const int block_y = blockIdx.y;
    const int start_m = block_x * BLOCK_DIM_M;
    const int start_n = block_y * BLOCK_DIM_N;

    __shared__ T a_sm[BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    __shared__ T b_sm[BLOCK_DIM_K][BLOCK_DIM_N + PAD_SIZE];
    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
    for (int k = 0; k < K; k += BLOCK_DIM_K) {
        load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, K, ld_thread_x, ld_thread_y,
            start_m, k, a, &a_sm[0][0]);
        load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, K, ld_thread_x, ld_thread_y,
            start_n, k, b, &b_sm[0][0]);
        __syncthreads();
        // if (k == 0 && block_x == 1 && block_y == 0 && thread_x == 0 && thread_y == 0) {
        //     for (int kk = 0 ; kk < BLOCK_DIM_K; kk++) {
        //         for (int mm = 0; mm < BLOCK_DIM_M + PAD_SIZE; mm++) {
        //             if (__half2float(a_sm[kk][mm]) == 0) {
        //                 continue;
        //             }
        //             printf("a_sm[%d][%d]: %lf ", kk, mm, __half2float(a_sm[kk][mm]));
        //             //printf("b_sm[%d][%d]: %lf", kk, mm, __half2float(b_sm[kk][mm]));
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        //     for (int kk = 0 ; kk < BLOCK_DIM_K; kk++) {
        //         for (int mm = 0; mm < BLOCK_DIM_M + PAD_SIZE; mm++) {
        //             if (__half2float(b_sm[kk][mm]) == 0) {
        //                 continue;
        //             }
        //             //printf("a_sm[%d][%d]: %lf", kk, mm, __half2float(a_sm[kk][mm]));
        //             printf("b_sm[%d][%d]: %lf ", kk, mm, __half2float(b_sm[kk][mm]));
        //         }
        //         printf("\n");
        //     }
        // }
        if constexpr (std::is_same_v<T, half>) {
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                cm_thread_x, cm_thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
        } else if constexpr (std::is_same_v<T, float>) {
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                cm_thread_x, cm_thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
        }
        __syncthreads();
    }

    __syncthreads();
    store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                  start_m, start_n, c,
                                                                  &c_reg[0]);
}

template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}

template<typename T>
void gemm_nn_cpu(int m, int n, int k, T *a, T *b, T *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                if constexpr (std::is_same_v<T, half>) {
                    float a_reg = __half2float(a[p * m + i]);
                    float b_reg = __half2float(b[p + j * k]);
                    sum += a_reg * b_reg;
                } else if constexpr (std::is_same_v<T, float>) {
                    sum += a[p * m + i] * b[p + j * k];
                }
            }
            if constexpr (std::is_same_v<T, half>) {
                c[j * m + i] = __float2half(sum); // C(i, j) in column-major
            } else {
                c[j * m + i] = sum;
            }
        }
    }
}

template<typename T>
void gemm_nt_cpu(int m, int n, int k, T *a, T *b, T *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                if constexpr (std::is_same_v<T, half>) {
                    float a_reg = __half2float(a[p * m + i]);
                    float b_reg = __half2float(b[p * n + j]);
                    sum += a_reg * b_reg;
                } else if constexpr (std::is_same_v<T, float>) {
                    sum += a[p * m + i] * b[p * n + j];
                }
            }
            if constexpr (std::is_same<T, half>::value) {
                c[j * m + i] = __float2half(sum); // C(i, j) in column-major
            } else {
                c[j * m + i] = sum;
            }
        }
    }
}

template<typename T>
void gemm_tt_cpu(int m, int n, int k, T *a, T *b, T *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                if constexpr (std::is_same_v<T, half>) {
                    float a_reg = __half2float(a[p + i * k]);
                    float b_reg = __half2float(b[p * n + j]);
                    sum += a_reg * b_reg;
                } else if constexpr (std::is_same_v<T, float>) {
                    sum += a[p + i * k] * b[p * n + j];
                }
            }
            if constexpr (std::is_same_v<T, half>) {
                c[j * m + i] = __float2half(sum);
            } else {
                c[j * m + i] = sum;
            }
        }
    }
}

template<typename T>
void gemm_tn_cpu(int m, int n, int k, T *a, T *b, T *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                if constexpr (std::is_same_v<T, half>) {
                    float a_reg = __half2float(a[p + i * k]);
                    float b_reg = __half2float(b[p + j * k]);
                    sum += a_reg * b_reg;
                } else if constexpr (std::is_same_v<T, float>) {
                    sum += a[p + i * k] * b[p + j * k];
                }
            }
            if constexpr (std::is_same_v<T, half>) {
                c[j * m + i] = __float2half(sum);
            } else {
                c[j * m + i] = sum;
            }
        }
    }
}

// void cutlass_sgemm_tf32_nn(int m, int n, int k,
//                         std::vector<float> &a_mat,
//                         std::vector<float> &b_mat,
//                         std::vector<float> &c_mat) {
//     using Gemm = cutlass::gemm::device::Gemm<
//         cutlass::tfloat32_t, // ElementA (TF32)
//         cutlass::layout::ColumnMajor, // LayoutA
//         cutlass::tfloat32_t, // ElementB (TF32)
//         cutlass::layout::ColumnMajor, // LayoutB
//         float, // ElementC (FP32 accumulator)
//         cutlass::layout::ColumnMajor // LayoutC
//     >;
//
//     // (M, N), row-major
//     std::vector<float> c_mat_gpu_result;
//     c_mat_gpu_result.resize(m * n);
//
//     float *a_mat_gpu = nullptr;
//     cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
//     cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
//     float *b_mat_gpu = nullptr;
//     cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
//     cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
//     float *c_mat_gpu = nullptr;
//     cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
//     cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
//
//
//     const auto *ptr_A = reinterpret_cast<const cutlass::tfloat32_t *>(a_mat_gpu);
//     const auto *ptr_B = reinterpret_cast<const cutlass::tfloat32_t *>(b_mat_gpu);
//     float *ptr_C = c_mat_gpu;
//
//     typename Gemm::Arguments args(
//         {m, n, k}, // GemmShape
//         {ptr_A, k}, // TensorRef A (data, lda)
//         {ptr_B, n}, // TensorRef B (data, ldb)
//         {ptr_C, n}, // TensorRef C (data, ldc)
//         {ptr_C, n}, // TensorRef D (output)
//         {1.0f, 0.0f} // alpha, beta
//     );
//
//     Gemm gemm_op;
//     size_t workspace_size = gemm_op.get_workspace_size(args);
//     void *workspace = nullptr;
//     if (workspace_size > 0) {
//         cudaMalloc(&workspace, workspace_size);
//     }
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     cutlass::Status status = gemm_op(args, workspace);
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
//                cudaMemcpyKind::cudaMemcpyDeviceToHost);
//     cudaFree(c_mat_gpu);
//     c_mat_gpu = nullptr;
//     cudaFree(a_mat_gpu);
//     a_mat_gpu = nullptr;
//     cudaFree(b_mat_gpu);
//     b_mat_gpu = nullptr;
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//
//     const long long flops = 2ll * m * n * k;
//     const double gflops = static_cast<double>(flops) / 1e9;
//     const double seconds = milliseconds / 1000.0;
//     const double gflops_per_sec = gflops / seconds;
//
//     printf("Matrix size: %d x %d x %d\n", m, n, k);
//     printf("Kernel time: %.4f ms\n", milliseconds);
//     printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
//     printf("cutlass Performance: %.2f GFLOPS/s\n", gflops_per_sec);
//
//     if (workspace) cudaFree(workspace);
//
//     if (status != cutlass::Status::kSuccess) {
//         std::cerr << "CUTLASS GEMM failed!" << std::endl;
//         return;
//     }
//
// #ifdef _DEBUG
//     //gemm_nn_cpu_col_major(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
//     printf("result c: \n");
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];
//
//             if (fabs(delta) > 0.01f) {
//                 printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
//                        c_mat[j * m + i]);
//                 return;
//             }
//             //printf("%lf ", delta);
//         }
//         //printf("\n");
//     }
//     printf("sgemm_wmma sucess !!: \n");
// #endif
// }
template<typename T>
void sgemm_nn_func(int m, int n, int k,
                   std::vector<T> &a_mat,
                   std::vector<T> &b_mat,
                   std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_K / VEC_DIM_K, 1);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nn_func<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<grid, block>>>(
        m, n, k, m, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;

    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 gemm nn func Performance: %.2f GFLOPS/s\n", gflops_per_sec);


#ifdef _DEBUG
    gemm_nn_cpu<T>(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

template<typename T>
void sgemm_nn_double_buffer(int m, int n, int k,
                            std::vector<T> &a_mat,
                            std::vector<T> &b_mat,
                            std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
    printf("grid x: %d, y: %d\n", grid.x, grid.y);
    printf("block x: %d, block y: %d\n", block.x, block.y);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nn_pipeline_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>
            <<<grid, block>>>(m, n, k, m, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;
    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 nn double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_nn_cpu<T>(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

template<typename T>
void sgemm_tt_func(int m, int n, int k,
                   std::vector<T> &a_mat,
                   std::vector<T> &b_mat,
                   std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tt_func<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<grid, block>>>(
        m, n, k, k, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;
    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 tt func Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_tt_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

template<typename T>
void sgemm_tt_double_buffer(int m, int n, int k,
                            std::vector<T> &a_mat,
                            std::vector<T> &b_mat,
                            std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tt_pipeline_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>
            <<<grid, block>>>(m, n, k, k, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;
    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 tt double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_tt_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

template<typename T>
void sgemm_tn_func(int m, int n, int k,
                   std::vector<T> &a_mat,
                   std::vector<T> &b_mat,
                   std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tn_func<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<grid, block>>>(
        m, n, k, k, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;
    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 tn func Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_tn_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

template<typename T>
void sgemm_tn_double_buffer(int m, int n, int k,
                            std::vector<T> &a_mat,
                            std::vector<T> &b_mat,
                            std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tn_pipeline_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>
            <<<grid, block>>>(m, n, k, k, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;
    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 tn double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_tn_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

template<typename T>
void sgemm_nt_func(int m, int n, int k,
                   std::vector<T> &a_mat,
                   std::vector<T> &b_mat,
                   std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);


    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nt_func<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<grid, block>>>(
        m, n, k, m, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;
    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 func nt Performance: %.2f GFLOPS/s\n", gflops_per_sec);


#ifdef _DEBUG
    gemm_nt_cpu<T>(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    //
    // printf("a_mat: \n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < k; j++) {
    //         printf("a[%d][%d]: %lf ", i, j, __half2float(a_mat[j * m + i]));
    //     }
    //     printf("\n");
    // }
    // //
    // printf("b_mat: \n");
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < k; j++) {
    //         printf("b[%d][%d]: %lf ", i, j, __half2float(b_mat[j * m + i]));
    //     }
    //     printf("\n");
    // }
    //
    // printf("result c: \n");
    // printf("c_mat_gpu: \n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("c[%d][%d]: %lf ", i, j, __half2float(c_mat_gpu_result[j * m + i]));
    //     }
    //     printf("\n");
    // }
    // printf("c_mat_cpu: \n");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("c[%d][%d]: %lf ", i, j, __half2float(c_mat[j * m + i]));
    //     }
    //     printf("\n");
    // }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

template<typename T>
void sgemm_nt_double_buffer(int m, int n, int k,
                            std::vector<T> &a_mat,
                            std::vector<T> &b_mat,
                            std::vector<T> &c_mat) {
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
    dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
    T *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(T));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(T));
    T *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(T));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(T), cudaMemcpyHostToDevice);
    T *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(T));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(T), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nt_pipeline_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>
            <<<grid, block>>>(m, n, k, m, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(T),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const long long flops = 2ll * m * n * k;
    const double gflops = static_cast<double>(flops) / 1e9;
    const double seconds = milliseconds / 1000.0;
    const double gflops_per_sec = gflops / seconds;
    printf("******************************************\n");
    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 nt double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_nt_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.001f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm sucess !!: \n");
    printf("******************************************\n");
#endif
}

int main(int argc, char *argv) {
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, 0);
    printf("device prop sharedMemPerBlock:%d \n", device_prop.sharedMemPerBlock);
    printf("device prop regsPerBlock: %d\n", device_prop.regsPerBlock);
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
#ifdef _DEBUG
    int dim = 1024;
    int m = dim;
    int n = dim;
    int k = dim;

    std::vector<T> a_mat;
    a_mat.resize(m * k);
    std::vector<T> b_mat;
    b_mat.resize(n * k);
    std::vector<T> c_mat;
    c_mat.resize(m * n);
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    PopulateVector<T>(a_mat, mt, dist);
    PopulateVector<T>(b_mat, mt, dist);

    //sgemm_nn_func<T>(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_nn_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    // // //
    sgemm_nt_func<T>(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_nt_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    // // // //
    // sgemm_tn_func<T>(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_tn_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    // // //
    // sgemm_tt_func<T>(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_tt_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    //
    //sgemm_wmma(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //sgemm_wmma_sm(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //
    //cutlass_sgemm_tf32(m, n, k, m, n, m, a_mat, b_mat, c_mat);
#else
    int dim = 4096;
    int m = dim;
    int n = dim;
    int k = dim;

    std::vector<T> a_mat;
    a_mat.resize(m * k);
    std::vector<T> b_mat;
    b_mat.resize(n * k);
    std::vector<T> c_mat;
    c_mat.resize(m * n);
    std::vector<T> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    PopulateVector<T>(a_mat, mt, dist);
    PopulateVector<T>(b_mat, mt, dist);

    //sgemm_nn_func<T>(m, n, k, a_mat, b_mat, c_mat);
    //sgemm_nn_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    //
    sgemm_nt_func<T>(m, n, k, a_mat, b_mat, c_mat);
    //sgemm_nt_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    // //
    // sgemm_tn_func<T>(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_tn_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    // //
    // sgemm_tt_func<T>(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_tt_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
    //
    //sgemm_wmma(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //sgemm_wmma_sm(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //
    //cutlass_sgemm_tf32(m, n, k, m, n, m, a_mat, b_mat, c_mat);
#endif


    return 0;
}
