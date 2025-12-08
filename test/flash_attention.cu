//
// Created by nkk on 2025/12/6.
//

#include <vector>
#include <random>
#include "cublas_v2.h"
#include "mma.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <math.h>
#include <cmath>
// #include "cutlass/gemm/device/gemm.h"
// #include "cutlass/gemm/device/gemm_universal_adapter.h"
// #include "cutlass/numeric_types.h"
// #include "cutlass/layout/matrix.h"
// #include <cuda_runtime.h>
// #include <iostream>

using T = half;
#if 0
constexpr int BYTES_PER_LOAD = 16; // 128-bit
constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD / sizeof(T);
constexpr int VEC_DIM_LOAD = 2 * ELEMENTS_PER_LOAD;
constexpr int WARP_SIZE = 32;
constexpr int THREAD_BLOCK_SIZE = 256;
constexpr int BLOCK_DIM_K = 16;
constexpr int VEC_DIM_N = 8;
constexpr int VEC_DIM_K = 1;
constexpr int VEC_DIM_M = 8;
constexpr int BLOCK_DIM_M = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_M;
constexpr int BLOCK_DIM_N = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_N;
constexpr int PAD_SIZE = 16; //BLOCK_DIM_K;
#else
constexpr int BYTES_PER_LOAD = 16; // 128-bit
constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD / sizeof(T);
constexpr int ACTUAL_ELEMENTS_PER_LOAD = ELEMENTS_PER_LOAD / 4;
constexpr int WARP_SIZE = 32;
constexpr int THREAD_BLOCK_SIZE = 256;
constexpr int BLOCK_DIM_K = 64;
constexpr int VEC_DIM_N = 4;
constexpr int VEC_DIM_K = 2;
constexpr int VEC_DIM_M = 16;
constexpr int BLOCK_DIM_M = 128; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_M;
constexpr int BLOCK_DIM_N = 128; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_N;
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
    if (count >= 1 && reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
        uint4 v = *reinterpret_cast<const uint4 *>(addr);
        const half *h = reinterpret_cast<const half *>(&v);
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = h[i];
    } else {
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
    }
}

template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
    const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int PAD_SIZE>
__device__ void load_tile_vec_row_major_n(const int ld, const int dim,
                                          const int thread_x, const int warp_id,
                                          const int start_m,
                                          const int k,
                                          const T *__restrict__ global_mem,
                                          T *sm) {
#pragma unroll
    for (int j = 0; j < VEC_DIM_LD; ++j) {
        const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
        for (int kk = 0; kk < VEC_DIM_K / ACTUAL_ELEMENTS_PER_LOAD; ++kk) {
            const int dim1 = k + thread_x * ACTUAL_ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                             ACTUAL_ELEMENTS_PER_LOAD;
            T val[ACTUAL_ELEMENTS_PER_LOAD] = {0};
            if (dim0_base < dim) {
                const int actual_load = min(ACTUAL_ELEMENTS_PER_LOAD, ld - dim1);
                if (actual_load > 0) {
                    load_vec<T>(&global_mem[dim0_base * ld + dim1], val, actual_load);
                }
            }
            int sm_col = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
            int sm_row_base = thread_x * ACTUAL_ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                              ACTUAL_ELEMENTS_PER_LOAD;
#pragma unroll
            for (int i = 0; i < ACTUAL_ELEMENTS_PER_LOAD; ++i) {
                sm[sm_col + (sm_row_base + i) * (BLOCK_DIM_LD + PAD_SIZE)] = val[i];
            }
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int PAD_SIZE>
__device__ void compute_tile_attention_gemm(const float scale, const int k_size, const int thread_x, const int warp_id,
                                            T *a_sm, T *b_sm,
                                            float *c_reg) {
#pragma unroll
    for (int kk = 0; kk < k_size; kk++) {
#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            float a_reg = 0.0f;
            if constexpr (std::is_same_v<T, half>) {
                a_reg = __half2float(
                    a_sm[(kk) * (BLOCK_DIM_M + PAD_SIZE) + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M]);
            } else if constexpr (std::is_same_v<T, float>) {
                a_reg = (a_sm[(kk) * (BLOCK_DIM_M + PAD_SIZE) + thread_x + mm *
                              BLOCK_DIM_M /
                              VEC_DIM_M]);
            }
#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                float b_reg = 0.0f;
                if constexpr (std::is_same_v<T, half>) {
                    b_reg = __half2float(
                        b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                } else if constexpr (std::is_same_v<T, float>) {
                    b_reg = (b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn *
                                  BLOCK_DIM_N / VEC_DIM_N]);
                }

                c_reg[mm * VEC_DIM_N + nn] += a_reg * b_reg;
            }
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int BLOCK_DIM_M>
__device__ void compute_tile_attention_softmax(const int N,const int thread_x,
                                               const int warp_id,
                                               const float scale,
                                               const int start_m,
                                               const int start_n,
                                               float *c_reg,
                                               float *max_value_global,
                                               float *sum_value_global,
                                               float *max_value,
                                               float *sum_value,
                                               float *new_max,
                                               float *new_sum) {
    const int base_n = start_n + thread_x;
    //const int valid_n = min(VEC_DIM_N, N - (base_n + VEC_DIM_N * 32));
    //printf("valid_n = %d\n", valid_n);
#pragma unroll
    for (int mm = 0; mm < VEC_DIM_M; mm++) {
        float old_max = max_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
        float old_sum = sum_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];

#pragma unroll
        for (int nn = 0; nn < VEC_DIM_N; nn++) {
            if ((nn * 32 +  base_n) < N) {
                c_reg[mm * VEC_DIM_N + nn] *= scale;
                max_value[mm] = c_reg[mm * VEC_DIM_N + nn] > max_value[mm] ? c_reg[mm * VEC_DIM_N + nn] : max_value[mm];
            }
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_value[mm] = fmaxf(max_value[mm], __shfl_xor_sync(0xffffffff, max_value[mm], offset, 32));
        }

#pragma unroll
        for (int nn = 0; nn < VEC_DIM_N; nn++) {
            if ((nn * 32 +  base_n) < N) {
                c_reg[mm * VEC_DIM_N + nn] = expf(c_reg[mm * VEC_DIM_N + nn] - max_value[mm]);
                sum_value[mm] += c_reg[mm * VEC_DIM_N + nn];
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_value[mm] += __shfl_xor_sync(0xffffffff, sum_value[mm], offset, 32);
        }
        new_max[mm] = fmaxf(max_value[mm], old_max);
        new_sum[mm] = sum_value[mm] * expf(max_value[mm] - new_max[mm]) + old_sum * expf(old_max - new_max[mm]);
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int BLOCK_DIM_M, const int BLOCK_DIM_K, const int PAD_SIZE>
__device__ void compute_tile_attention_pv(const int N, const int v_ld,
                                          const int thread_x,
                                          const int warp_id,
                                          const int start_m,
                                          const int start_n,
                                          const T *v_sm,
                                          float *c_reg,
                                          float *old_max_value,
                                          float *old_sum_value,
                                          float *current_max_value,
                                          float *current_sum_value,
                                          float *new_max_value,
                                          float *new_sum_value,
                                          float *output) {
    const int base_n = start_n + thread_x;
    const int valid_n = min(VEC_DIM_N, N - base_n);
    for (int kk = 0; kk < BLOCK_DIM_K; kk++) {
#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            float acc = 0.0f;
#pragma unroll
            for (int nn = 0; nn < valid_n; nn++) {
                float v_reg = 0.0f;
                if constexpr (std::is_same_v<T, half>) {
                    v_reg = __half2float(
                        v_sm[kk * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                } else if constexpr (std::is_same_v<T, float>) {
                    v_reg = v_sm[kk * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N];
                }
                acc += v_reg * c_reg[mm * VEC_DIM_N + nn];
                // if (kk == 0 && thread_x == 0 && warp_id == 0 && blockIdx.x == 0) {
                //     printf("mm: %d, nn: %d, v_reg: %lf, c_reg[%d]: %lf, acc: %lf \n", mm, nn, v_reg,
                //            mm * VEC_DIM_N + nn, c_reg[mm * VEC_DIM_N + nn], acc);
                // }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                acc += __shfl_xor_sync(0xffffffff, acc, offset, 32);
            }
            // if (kk == 0 && thread_x == 0 && blockIdx.x == 0) {
            //     printf("warp_id: %d, mm: %d,acc: %lf new_sum: %lf \n", warp_id, mm, acc, new_sum_value[mm]);
            // }
            float old_max = old_max_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
            //if ((start_m + BLOCK_DIM_M) >= N) {
                output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] =
                (output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] * expf(
                     old_max - new_max_value[mm]) +
                 acc * expf(current_max_value[mm] - new_max_value[mm])) / new_sum_value[mm];
            // } else {
            //     output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] =
            //             output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] * expf(
            //                 old_max - new_max_value[mm]) +
            //             acc * expf(current_max_value[mm] - new_max_value[mm]);
            // }

            old_max_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_max_value[mm];
            old_sum_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_sum_value[mm];
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__global__ void flash_attention_cuda(const int M, const int N, const int D,
                                     const int q_ld, const int k_ld, const int v_ld,
                                     const float scale,
                                     const int start_n,
                                     const int n_block_size,
                                     const T *__restrict__ q_global,
                                     const T *__restrict__ k_global,
                                     const T *__restrict__ v_global,
                                     float *__restrict__ max_value_global,
                                     float *__restrict__ sum_value_global,
                                     float *out_put) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int ld_thread_block_n = BLOCK_DIM_K / VEC_DIM_K;
    const int thread_x = thread_id % ld_thread_block_n;
    const int warp_id = thread_id / ld_thread_block_n;
    const int warp_group_id = thread_id / 128;

    const int block_x = blockIdx.x;
    const int start_m = block_x * BLOCK_DIM_M;

    __shared__ T sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
    float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

    for (int d = 0; d < D; d += BLOCK_DIM_K) {
        //一个k block相当于一个head, head间独立
        //加载当前q块;
        load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
            q_ld, M, thread_x, warp_id, start_m, d,
            q_global, &sm[0][0][0]);
        load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
            k_ld, N, thread_x, warp_id, start_n, d,
            k_global, &sm[1][0][0]);

        __syncthreads();
        // if (d == 0 && thread_x == 0 && warp_id == 0 && block_x == 0) {
        //     for (int i = 0; i < BLOCK_DIM_K; ++i) {
        //         for (int j = 0; j < BLOCK_DIM_M; ++j) {
        //             printf("q_sm[%d][%d]: %lf ", i, j, __half2float(sm[0][i][j]));
        //         }
        //         printf("\n");
        //     }
        //
        //     for (int i = 0; i < BLOCK_DIM_K; ++i) {
        //         for (int j = 0; j < BLOCK_DIM_N; ++j) {
        //             printf("k_sm[%d][%d]: %lf ", i, j, __half2float(sm[1][i][j]));
        //         }
        //         printf("\n");
        //     }
        // }

        const int k_size = min(BLOCK_DIM_K, D - d);
        compute_tile_attention_gemm<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
            scale, k_size, thread_x, warp_id, &sm[0][0][0], &sm[1][0][0], c_reg);

        __syncthreads();
        // if (start_m == 0 && start_n == 0) {
        //     for (int mm = 0; mm < VEC_DIM_M; ++mm) {
        //         //printf("max: %f\n", max_value[mm]);
        //         for (int nn = 0; nn < VEC_DIM_N; ++nn) {
        //             if (c_reg[mm * VEC_DIM_N + nn] != 0) {
        //                 printf("warp_id: %d, thread_x: %d, c_reg[%d][%d]: %f \n",warp_id, thread_x,mm, nn, c_reg[mm * VEC_DIM_N + nn]);
        //             }
        //         }
        //         //printf("\n");
        //     }
        // }

        //softmax;
        float max_value[VEC_DIM_M] = {-1e20f};
        float sum_value[VEC_DIM_M] = {0.0f};
        float new_max_value[VEC_DIM_M] = {-1e20f};
        float new_sum_value[VEC_DIM_M] = {0.0f};
        compute_tile_attention_softmax<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M>(
            N, thread_x, warp_id, scale, start_m,start_n,
            &c_reg[0], max_value_global, sum_value_global, &max_value[0], &sum_value[0], &new_max_value[0],
            &new_sum_value[0]);
        __syncthreads();
        // if (start_m == 0 && start_n == 0 && warp_id == 0) {
        //     for (int mm = 0; mm < VEC_DIM_M; ++mm) {
        //         //printf("warp_id: %d, max: %f, sum: %lf \n", warp_id, max_value[mm], sum_value[mm]);
        //
        //         for (int nn = 0; nn < VEC_DIM_N; ++nn) {
        //             if (mm == 0) {
        //                 printf(
        //                     "mm： %d, nn: %d, warp_id: %d, thread_x: %d, c_reg[%d][%d]: %10f, max: %f, sum: %lf new_max: %f, new_sum: %f \n",
        //                     mm, nn, warp_id, thread_x, mm, nn, c_reg[mm * VEC_DIM_N + nn], max_value[mm], sum_value[mm],
        //                     new_max_value[mm], new_sum_value[mm]);
        //             }
        //         }
        //         //printf("\n");
        //     }
        // }
        load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
            v_ld, N, thread_x, warp_id, start_n, d,
            v_global, &sm[1][0][0]);
        __syncthreads();
        // if (d == 0 && thread_x == 0 && warp_id == 0 && block_x == 0) {
        //     for (int i = 0; i < BLOCK_DIM_K; ++i) {
        //         for (int j = 0; j < BLOCK_DIM_N; ++j) {
        //             printf("v_sm[%d][%d]: %lf ", i, j, __half2float(sm[1][i][j]));
        //         }
        //         printf("\n");
        //     }
        // }
        //pv;
        compute_tile_attention_pv<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
            N, v_ld, thread_x, warp_id, start_m,start_n,
            &sm[1][0][0], &c_reg[0], max_value_global, sum_value_global,
            &max_value[0], &sum_value[0], &new_max_value[0],
            &new_sum_value[0], out_put);
    }
}

// template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
//     const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
// __global__ void flash_attention_cuda(const int M, const int N, const int D,
//                                      const int q_ld, const int k_ld, const int v_ld,
//                                      const float scale,
//                                      const int start_n,
//                                      const int n_block_size,
//                                      const T *__restrict__ q_global,
//                                      const T *__restrict__ k_global,
//                                      const T *__restrict__ v_global,
//                                      T *__restrict__ max_value_global,
//                                      T *__restrict__ sum_value_global,
//                                      float *out_put) {
//     const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
//     const int ld_thread_block_n = BLOCK_DIM_K / VEC_DIM_K;
//     const int thread_x = thread_id % ld_thread_block_n;
//     const int warp_id = thread_id / ld_thread_block_n;
//     const int warp_group_id = thread_id / 128;
//
//     const int block_x = blockIdx.x;
//     const int start_m = block_x * BLOCK_DIM_M;
//
//
//     __shared__ T sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
//     float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};
//
//     for (int k = 0; k < D; k += BLOCK_DIM_K) {
//         //一个k block相当于一个head, head间独立
//         //加载当前q块;
//         load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
//             q_ld, M, thread_x, warp_id, start_m, k,
//             q_global, &sm[0][0][0]);
//         load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
//             k_ld, N, thread_x, warp_id, start_n, k,
//             k_global, &sm[1][0][0]);
//
//         __syncthreads();
//         if (k == 0 && thread_x == 0 && warp_id == 0 && block_x == 0 ) {
//             for (int i = 0; i < BLOCK_DIM_K; ++i) {
//                 for (int j = 0; j < BLOCK_DIM_M; ++j) {
//                     printf("q_sm[%d][%d]: %lf ", i, j, __half2float(sm[0][i][j]));
//                 }
//                 printf("\n");
//             }
//
//             for (int i = 0; i < BLOCK_DIM_K; ++i) {
//                 for (int j = 0; j < BLOCK_DIM_N; ++j) {
//                     printf("k_sm[%d][%d]: %lf ", i, j, __half2float(sm[1][i][j]));
//                 }
//                 printf("\n");
//             }
//         }
//         if (warp_group_id == 0) {
//             const int k_size = min(BLOCK_DIM_K, D - k);
//             compute_tile_attention_gemm<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
//                 scale, k_size, thread_x, warp_id, &sm[0][0][0], &sm[1][0][0], c_reg);
//
//             __barrier_sync(0);
//             if (start_m == 0 && start_n == 0 && warp_id == 0 && thread_x == 0) {
//                 for (int mm = 0; mm < VEC_DIM_M; ++mm) {
//                     //printf("max: %f\n", max_value[mm]);
//                     for (int nn = 0; nn < VEC_DIM_N; ++nn) {
//                         printf("c_reg[%d][%d]: %f ", mm, nn, c_reg[mm * VEC_DIM_N + nn]);
//                     }
//                     printf("\n");
//                 }
//             }
//
//             //softmax;
//             float max_value[VEC_DIM_M] = {-1e20f};
//             float sum_value[VEC_DIM_M] = {0.0f};
//             float new_max_value[VEC_DIM_M] = {-1e20f};
//             float new_sum_value[VEC_DIM_M] = {0.0f};
//             compute_tile_attention_softmax<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M>(
//                 thread_x, warp_id, scale, start_m,
//                 &c_reg[0], max_value_global, sum_value_global, &max_value[0], &sum_value[0], &new_max_value[0],
//                 &new_sum_value[0]);
//
//             load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
//                 v_ld, N, thread_x, warp_id, start_n, k,
//                 v_global, &sm[1][0][0]);
//             __syncthreads();
//             //pv;
//             compute_tile_attention_pv<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
//                 v_ld, thread_x, warp_id, start_m,
//                 &sm[2][0][0], &c_reg[0], max_value_global, sum_value_global,
//                 &max_value[0], &sum_value[0], &new_max_value[0],
//                 &new_sum_value[0], out_put);
//         } else if (warp_group_id == 1) {
//             __barrier_sync(0);
//             const int k_size = min(BLOCK_DIM_K, D - k);
//             compute_tile_attention_gemm<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
//                 scale, k_size, thread_x, warp_id, &sm[0][0][0], &sm[1][0][0], c_reg);
//
//             //softmax;
//             float max_value[VEC_DIM_M] = {-1e20f};
//             float sum_value[VEC_DIM_M] = {0.0f};
//             float new_max_value[VEC_DIM_M] = {-1e20f};
//             float new_sum_value[VEC_DIM_M] = {0.0f};
//             compute_tile_attention_softmax<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M>(
//                 thread_x, warp_id, scale, start_m,
//                 &c_reg[0], max_value_global, sum_value_global, &max_value[0], &sum_value[0], &new_max_value[0],
//                 &new_sum_value[0]);
//
//             load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
//                 v_ld, N, thread_x, warp_id, start_n, k,
//                 v_global, &sm[1][0][0]);
//             __syncthreads();
//             //pv;
//             compute_tile_attention_pv<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
//                 v_ld, thread_x, warp_id, start_m,
//                 &sm[2][0][0], &c_reg[0], max_value_global, sum_value_global,
//                 &max_value[0], &sum_value[0], &new_max_value[0],
//                 &new_sum_value[0], out_put);
//         }
//     }
// }
template<typename T>
inline float to_float(T val) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, float>) {
        return val;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

template<typename T>
std::vector<float> flash_attention_cpu(
    int m, int n, int k,
    const std::vector<T> &q_mat,
    const std::vector<T> &k_mat,
    const std::vector<T> &v_mat,
    int block_size_m = 128,
    int block_size_n = 128) {
    if (q_mat.size() != static_cast<size_t>(m * k) ||
        k_mat.size() != static_cast<size_t>(n * k) ||
        v_mat.size() != static_cast<size_t>(n * k)) {
        return std::vector<float>();
    }

    const float scaling = 1.0f / std::sqrt(static_cast<float>(k));
    std::vector<float> output(m * k, 0.0f);

    for (int m_start = 0; m_start < m; m_start += block_size_m) {
        int m_end = std::min(m_start + block_size_m, m);
        int current_m = m_end - m_start;

        std::vector<float> running_max(current_m, -1e20f); // m_i
        std::vector<float> running_sum(current_m, 0.0f); // l_i

        for (int n_start = 0; n_start < n; n_start += block_size_n) {
            int n_end = std::min(n_start + block_size_n, n);
            int current_n = n_end - n_start;

            std::vector<std::vector<float> > local_logits(current_m, std::vector<float>(current_n, 0.0f));
            for (int im = 0; im < current_m; ++im) {
                int i = m_start + im;
                for (int j_idx = 0; j_idx < current_n; ++j_idx) {
                    int j = n_start + j_idx;
                    float dot = 0.0f;
                    for (int l = 0; l < k; ++l) {
                        dot += to_float(q_mat[i * k + l]) * to_float(k_mat[j * k + l]);
                    }
                    local_logits[im][j_idx] = dot * scaling;
                    //printf("cpu c_reg*scale[%d][%d]: %lf ", im, j_idx, local_logits[im][j_idx]);
                }
                //printf("\n");
            }

            std::vector<float> local_max(current_m);
            std::vector<float> local_sum(current_m);
            std::vector<std::vector<float> > local_exps(current_m, std::vector<float>(current_n));

            for (int im = 0; im < current_m; ++im) {
                // Find local max
                local_max[im] = *std::max_element(local_logits[im].begin(), local_logits[im].end());

                // Compute exps and sum
                local_sum[im] = 0.0f;
                for (int j_idx = 0; j_idx < current_n; ++j_idx) {
                    local_exps[im][j_idx] = float(::expf(local_logits[im][j_idx] - local_max[im]));
                    local_sum[im] += local_exps[im][j_idx];
                    //printf("cpu c_reg[%d][%d]: %lf ", im, j_idx, local_exps[im][j_idx]);
                }
                //printf("\n");
            }

            for (int im = 0; im < current_m; ++im) {
                int i = m_start + im;

                float old_max = running_max[im];
                float new_max = std::max(old_max, local_max[im]);
                float exp_old = ::expf(old_max - new_max);
                float exp_local = ::expf(local_max[im] - new_max);
                float new_sum = running_sum[im] * exp_old + local_sum[im] * exp_local;


                for (int l = 0; l < k; ++l) {
                    float acc = 0.0f;
                    for (int j_idx = 0; j_idx < current_n; ++j_idx) {
                        int j = n_start + j_idx;
                        acc += local_exps[im][j_idx] * to_float(v_mat[j * k + l]);
                    }

                    float corrected_acc = (running_sum[im] * exp_old * output[i * k + l] + acc * exp_local) / new_sum;
                    output[i * k + l] = corrected_acc;
                }

                running_max[im] = new_max;
                running_sum[im] = new_sum;
            }
        }
    }
    return output;
}

template<typename T>
void flash_attention(int M, int N, int D,
                     const std::vector<T> &q_mat,
                     const std::vector<T> &k_mat,
                     const std::vector<T> &v_mat) {
    T *q_gpu = nullptr;
    cudaMalloc((void **) &q_gpu, sizeof(T) * M * D);
    cudaMemcpy(q_gpu, q_mat.data(), sizeof(T) * M * D, cudaMemcpyHostToDevice);
    T *k_gpu = nullptr;
    cudaMalloc((void **) &k_gpu, sizeof(T) * N * D);
    cudaMemcpy(k_gpu, k_mat.data(), sizeof(T) * N * D, cudaMemcpyHostToDevice);
    T *v_gpu = nullptr;
    cudaMalloc((void **) &v_gpu, sizeof(T) * N * D);
    cudaMemcpy(v_gpu, v_mat.data(), sizeof(T) * N * D, cudaMemcpyHostToDevice);
    float *max_value = nullptr;
    cudaMalloc((void **) &max_value, sizeof(float) * M);
    cudaMemset(max_value, 0, sizeof(float) * M);
    float *sum_value = nullptr;
    cudaMalloc((void **) &sum_value, sizeof(float) * M);
    cudaMemset(sum_value, 0, sizeof(float) * M);

    float *output = nullptr;
    cudaMalloc((void **) &output, sizeof(float) * M * D);
    cudaMemset(output, 0, sizeof(float) * M * D);
    const int block_num = max(N / BLOCK_DIM_N, 1);
    for (int j = 0; j < block_num; ++j) {
        int j_start = j * BLOCK_DIM_N;
        const int actual_Bc = (j_start + BLOCK_DIM_N > N) ? (N - j_start) : BLOCK_DIM_N;

        dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, 1, 1);
        dim3 block(256);
        flash_attention_cuda<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<grid
                , block>>>(
                    M, N, D,
                    D, D, D,
                    (1.0f / std::sqrt(static_cast<float>(D))),
                    j_start,
                    actual_Bc,
                    q_gpu, k_gpu, v_gpu,
                    max_value, sum_value, output);
        cudaDeviceSynchronize();
    }
    std::vector<float> output_cpu;
    output_cpu.resize(M * D);
    cudaMemcpy(output_cpu.data(), output, sizeof(float) * M * D, cudaMemcpyDeviceToHost);

    cudaFree(q_gpu);
    cudaFree(k_gpu);
    cudaFree(v_gpu);
    cudaFree(max_value);
    cudaFree(sum_value);
    cudaFree(output);
#ifdef _DEBUG
    // printf("q_mat: \n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < D; j++) {
    //         printf("q[%d][%d]: %lf ", i, j, __half2float(q_mat[i * D + j]));
    //     }
    //     printf("\n");
    // }
    // //
    // printf("k_mat: \n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < D; j++) {
    //         printf("k[%d][%d]: %lf ", i, j, __half2float(k_mat[i * D + j]));
    //     }
    //     printf("\n");
    // }
    // printf("v_mat: \n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < D; j++) {
    //         printf("v[%d][%d]: %lf ", i, j, __half2float(v_mat[i * D + j]));
    //     }
    //     printf("\n");
    // }

    std::vector<float> cpu_result = flash_attention_cpu(M, N, D, q_mat, k_mat, v_mat);
    // printf("cpu_result: \n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < D; j++) {
    //         printf("v[%d][%d]: %lf ", i, j, __half2float(cpu_result[i * D + j]));
    //     }
    //     printf("\n");
    // }
    // printf("gpu_result: \n");
    // for (int i = 0; i < M; i++) {
    //     for (int j = 0; j < D; j++) {
    //         printf("v[%d][%d]: %lf ", i, j, __half2float(output_cpu[i * D + j]));
    //     }
    //     printf("\n");
    // }

    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < D; ++i) {
            float delta = cpu_result[j * D + i] - output_cpu[j * D + i];
            if (fabs(delta) > 0.01f) {
                printf("error : %f, m: %d, d: %d\n", delta, j, i);
                return;
            }
        }
    }
    printf("success\n");
#endif
}

template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
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
    int dim = 256;
    int m = dim;
    int n = dim;
    int k = 64;

    std::vector<T> a_mat;
    a_mat.resize(m * k);
    std::vector<T> b_mat;
    b_mat.resize(n * k);
    std::vector<T> c_mat;
    c_mat.resize(n * k);

    PopulateVector<T>(a_mat, mt, dist);
    PopulateVector<T>(b_mat, mt, dist);
    PopulateVector<T>(c_mat, mt, dist);
    flash_attention<T>(m, n, k, a_mat, b_mat, c_mat);

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
    sgemm_nt_double_buffer<T>(m, n, k, a_mat, b_mat, c_mat);
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
