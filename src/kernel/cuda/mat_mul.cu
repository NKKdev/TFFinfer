//
// Created by nkk on 2025/12/9.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {
    template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int
        PAD_SIZE, const int ELEMENTS_PER_LOAD>
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

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int PAD_SIZE>
    __device__ void compute_tile(const int k_size, const int thread_x, const int thread_y,
                                 T *a_sm, T *b_sm,
                                 float *c_reg) {
#pragma unroll
        for (int kk = 0; kk < k_size; kk++) {
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                float a_reg = 0.0f;
                if constexpr (std::is_same_v<T, half>) {
                    a_reg = __half2float(
                        a_sm[(kk) * (BLOCK_DIM_M + PAD_SIZE) + thread_x + mm *
                             BLOCK_DIM_M /
                             VEC_DIM_M]);
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
                            b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_y + nn *
                                 BLOCK_DIM_N /
                                 VEC_DIM_N]);
                    } else if constexpr (std::is_same_v<T, float>) {
                        b_reg = (b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_y + nn *
                                      BLOCK_DIM_N / VEC_DIM_N]);
                    }

                    c_reg[nn * VEC_DIM_M + mm] += a_reg * b_reg;
                }
            }
        }
    }
    //
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

    template<typename T, const int VEC_DIM_LOAD, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_nn_pipeline_double_buffer_vec(
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
        load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(a_ld, K, ld_thread_x, ld_thread_y,
            start_m, 0, a, &a_sm[flip_flag][0][0]);
        load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, ld_thread_x, ld_thread_y,
                                                                                 start_n, 0, b, &b_sm[flip_flag][0][0]);
        __syncthreads();


        for (int k = 0; k <= K; k += BLOCK_DIM_K) {

            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            const int next_k = k + BLOCK_DIM_K;
            if (next_k < K) {
                //load 下一块数据到sm;
                load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(
                    a_ld, K, ld_thread_x, ld_thread_y,
                    start_m, next_k, a, &a_sm[!flip_flag][0][0]);
                load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, ld_thread_x, ld_thread_y,
                    start_n, next_k, b, &b_sm[!flip_flag][0][0]);
            }
            __syncthreads();
            flip_flag ^= 1;
        }

        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                      start_m, start_n, c,
                                                                      &c_reg[0]);
    }

    template<typename T, const int VEC_DIM_LOAD, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_nn_func_vec(int M, int N, int K, int a_ld, int b_ld, int c_ld,
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
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(
                a_ld, K, ld_thread_x, ld_thread_y,
                start_m, k, a, &a_sm[0][0]);
            load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, ld_thread_x, ld_thread_y,
                start_n, k, b, &b_sm[0][0]);
            __syncthreads();
            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }

        __syncthreads();
        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                      start_m, start_n, c,
                                                                      &c_reg[0]);
    }

    template<typename T, const int VEC_DIM_LOAD, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_tt_pipeline_double_buffer_vec(
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
        __shared__ T b_sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];

        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        int flip_flag = 0;
        load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, ld_thread_x, ld_thread_y,
                                                                                 start_m, 0, a, &a_sm[flip_flag][0][0]);
        load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(b_ld, K, ld_thread_x, ld_thread_y,
            start_n, 0, b, &b_sm[flip_flag][0][0]);
        __syncthreads();


        for (int k = 0; k <= K; k += BLOCK_DIM_K) {

            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            const int next_k = k + BLOCK_DIM_K;
            if (next_k < K) {
                //load 下一块数据到sm;
                load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, ld_thread_x, ld_thread_y,
                    start_m, next_k, a, &a_sm[!flip_flag][0][0]);
                load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(
                    b_ld, K, ld_thread_x, ld_thread_y,
                    start_n, next_k, b, &b_sm[!flip_flag][0][0]);
            }
            __syncthreads();
            flip_flag ^= 1;
        }
        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                      start_m, start_n, c,
                                                                      &c_reg[0]);
    }

    template<typename T, const int VEC_DIM_LOAD, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_tt_func_vec(int M, int N, int K, int a_ld, int b_ld, int c_ld,
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
            load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, ld_thread_x, ld_thread_y,
                start_m, k, a, &a_sm[0][0]);
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(b_ld, K, ld_thread_x, ld_thread_y,
                start_n, k, b, &b_sm[0][0]);
            __syncthreads();
            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }

        __syncthreads();
        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                      start_m, start_n, c,
                                                                      &c_reg[0]);
    }

    template<typename T, const int VEC_DIM_LOAD, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_tn_pipeline_double_buffer_vec(
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
        load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, ld_thread_x, ld_thread_y,
                                                                                 start_m, 0, a, &a_sm[flip_flag][0][0]);
        load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, ld_thread_x, ld_thread_y,
                                                                                 start_n, 0, b, &b_sm[flip_flag][0][0]);
        __syncthreads();


        for (int k = 0; k < K; k += BLOCK_DIM_K) {

            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            const int next_k = k + BLOCK_DIM_K;
            if (next_k < K) {
                //load 下一块数据到sm;
                load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, ld_thread_x, ld_thread_y,
                    start_m, next_k, a, &a_sm[!flip_flag][0][0]);
                load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, ld_thread_x, ld_thread_y,
                    start_n, next_k, b, &b_sm[!flip_flag][0][0]);
            }
            __syncthreads();
            flip_flag ^= 1;
        }
        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                      start_m, start_n, c,
                                                                      &c_reg[0]);
    }

    template<typename T, const int VEC_DIM_LOAD, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_tn_func_vec(int M, int N, int K, int a_ld, int b_ld, int c_ld,
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
            load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(a_ld, M, ld_thread_x, ld_thread_y,
                start_m, k, a, &a_sm[0][0]);
            load_tile_t<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(b_ld, N, ld_thread_x, ld_thread_y,
                start_n, k, b, &b_sm[0][0]);
            __syncthreads();
            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }

        __syncthreads();
        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                      start_m, start_n, c,
                                                                      &c_reg[0]);
    }

    template<typename T,const int VEC_DIM_LOAD,  const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_nt_pipeline_double_buffer_vec(
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

        int flip_flag = 0;
        load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(a_ld, K, ld_thread_x, ld_thread_y,
            start_m, 0, a, &a_sm[flip_flag][0][0]);
        load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(b_ld, K, ld_thread_x, ld_thread_y,
            start_n, 0, b, &b_sm[flip_flag][0][0]);
        __syncthreads();

        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        for (int k = 0; k <= K; k += BLOCK_DIM_K) {
            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],
                &c_reg[0]);

            const int next_k = k + BLOCK_DIM_K;
            if (next_k < K) {
                //load 下一块数据到sm;
                load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(
                    a_ld, K, ld_thread_x, ld_thread_y,
                    start_m, next_k, a, &a_sm[!flip_flag][0][0]);
                load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(
                    b_ld, K, ld_thread_x, ld_thread_y,
                    start_n, next_k, b, &b_sm[!flip_flag][0][0]);
            }
            __syncthreads();
            flip_flag ^= 1;
        }

        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                                                                      start_m, start_n, c,
                                                                      &c_reg[0]);
    }


    template<typename T, const int VEC_DIM_LOAD, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void gemm_nt_func_vec(int M, int N, int K, int a_ld, int b_ld, int c_ld,
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
        //#pragma unroll
        for (int k = 0; k < K; k += BLOCK_DIM_K) {
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(
                a_ld, K, ld_thread_x, ld_thread_y,
                start_m, k, a, &a_sm[0][0]);
            load_tile_n_vec<T, VEC_DIM_LOAD, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, VEC_DIM_LOAD>(
                b_ld, K, ld_thread_x, ld_thread_y,
                start_n, k, b, &b_sm[0][0]);
            __syncthreads();

            const int k_size = min(BLOCK_DIM_K, K - k);
            compute_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                k_size, cm_thread_x, cm_thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }
        store_tile<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M,
            BLOCK_DIM_N>(a_ld, b_ld, c_ld, cm_thread_x, cm_thread_y,
                         start_m, start_n, c,
                         &c_reg[0]);
    }

    template<typename T>
    static void gemm_kernel_cuda(const tff::core::graph::MatMulTransType &trans_type,
                                 std::vector<std::shared_ptr<tff::core::memory::Tensor> > &src,
                                 std::vector<std::shared_ptr<tff::core::memory::Tensor> > &dst,
                                 std::shared_ptr<core::runtime::LLMWeightMemManager> &mem_buffer_manager_ptr) {
        auto &input_tensor_a = *src.begin();
        auto &input_tensor_b = *src.rbegin();
        auto &output_tensor = *dst.begin();
        if (input_tensor_a->get_buffer() == nullptr) {
            tff::log::Logger::error("Input tensor A is null.");
            return;
        }
        if (input_tensor_b->get_buffer() == nullptr) {
            tff::log::Logger::error("Input tensor B is null.");
            return;
        }

        auto &a_shape = input_tensor_a->get_shape();
        int D = 0;
        int S = 0;
        int B = 0;
        if (a_shape.size() == 2) {
            //单个batch
            D = a_shape[0]; //D
            S = a_shape[1]; //S
            B = 1; //B
        } else if (a_shape.size() == 3) {
            //多个batch
            D = a_shape[0]; //D
            S = a_shape[1]; //S
            B = a_shape[2]; //B
        }


        auto mem_buffer = mem_buffer_manager_ptr->get_gpu_memory();
        if (mem_buffer.second == nullptr) {
            tff::log::Logger::error("rms_norm_kernel_cuda: mem_buffer_manager_ptr is nullptr!");
            return;
        }
        output_tensor->set_buffer_data(mem_buffer.second, output_tensor->get_bytes(), mem_buffer.first);


        constexpr int ELEMENTS_PER_LOAD = 4;
        constexpr int VEC_DIM_LOAD = ELEMENTS_PER_LOAD;
        constexpr int THREAD_BLOCK_SIZE = 256;
        constexpr int BLOCK_DIM_K = 16;
        constexpr int VEC_DIM_N = 8;
        constexpr int VEC_DIM_K = 2;
        constexpr int VEC_DIM_M = 8;
        constexpr int BLOCK_DIM_M = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_LOAD;
        constexpr int BLOCK_DIM_N = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_LOAD;
        constexpr int PAD_SIZE = 1; //BLOCK_DIM_K;

        dim3 grid((S + BLOCK_DIM_M - 1) / BLOCK_DIM_M,
              (D + BLOCK_DIM_N - 1) / BLOCK_DIM_N,
              1);
        dim3 block(BLOCK_DIM_N / VEC_DIM_N, BLOCK_DIM_M / VEC_DIM_M, 1);
        switch (trans_type) {
            case tff::core::graph::MatMulTransType::TFF_TT:
                gemm_tt_pipeline_double_buffer_vec<float, VEC_DIM_LOAD, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<
                        grid,
                        block>>>(S, D, D, S, D, D, static_cast<float *>(input_tensor_a->get_buffer()->ptr()),
                                 static_cast<float *>(input_tensor_b->get_buffer()->ptr()),
                                 static_cast<float *>(output_tensor->get_buffer()->ptr()));
                break;
            case tff::core::graph::MatMulTransType::TFF_NT:
                gemm_nt_pipeline_double_buffer_vec<float, VEC_DIM_LOAD, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<
                        grid,
                        block>>>(S, D, D, S, D, D, static_cast<float *>(input_tensor_a->get_buffer()->ptr()),
                                 static_cast<float *>(input_tensor_b->get_buffer()->ptr()),
                                 static_cast<float *>(output_tensor->get_buffer()->ptr()));
                break;
            case tff::core::graph::MatMulTransType::TFF_NN:
                gemm_nn_pipeline_double_buffer_vec<float, VEC_DIM_LOAD, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<
                        grid,
                        block>>>(S, D, D, S, D, D, static_cast<float *>(input_tensor_a->get_buffer()->ptr()),
                                 static_cast<float *>(input_tensor_b->get_buffer()->ptr()),
                                 static_cast<float *>(output_tensor->get_buffer()->ptr()));
                break;
            case tff::core::graph::MatMulTransType::TFF_TN:
                gemm_tn_pipeline_double_buffer_vec<float, VEC_DIM_LOAD, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE><<<
                        grid,
                        block>>>(S, D, D, S, D, D, static_cast<float *>(input_tensor_a->get_buffer()->ptr()),
                                 static_cast<float *>(input_tensor_b->get_buffer()->ptr()),
                                 static_cast<float *>(output_tensor->get_buffer()->ptr()));
                break;
        }
    }

    template<typename T>
    void tff::kernel::XGemm<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), XGemm<T>::get_op_name().c_str());
        auto trans_type = get_param_value<tff::core::graph::MatMulTransType>(1, para_ptr);
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            2, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            3, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(4, para_ptr);

        if (input_tensors.size() != 2) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        if (output_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        //
        gemm_kernel_cuda<T>(trans_type, input_tensors, output_tensors, mem_buffer_manager_ptr);
    }
    template<typename T>
    std::string tff::kernel::XGemm<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MUL_MAT);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();
        return name;
    }
    template class tff::kernel::XGemm<float>;
    template class tff::kernel::XGemm<half>;
    template class tff::kernel::XGemm<Q8_0>;
    REGISTER_OP_OBJECT(XGemm, float);
    REGISTER_OP_OBJECT(XGemm, half);
    REGISTER_OP_OBJECT(XGemm, Q8_0);
}