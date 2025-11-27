//
// Created by nkk on 2025/11/3.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<const int VEC_DIM_M, const int VEC_DIM_N, const int M_DIM_SIZE, const int BLOCK_PAD_SIZE>
    __device__ void load_tile_n(const int ld, const int dim,
                                const int thread_x, const int thread_y,
                                const int start_m,
                                const int k,
                                const float *__restrict__ global_mem,
                                float *sm) {
#pragma unroll
        for (int j = 0; j < VEC_DIM_M; j++) {
            int dim0 = start_m + thread_x + j * M_DIM_SIZE;
            int dim1 = k + thread_y;
            float val = 0.0f;
            if (dim1 < dim && dim0 < ld) {
                val = __ldg(&global_mem[dim1 * ld + dim0]);
            }
            sm[thread_y * BLOCK_PAD_SIZE + thread_x + j * M_DIM_SIZE] = val;
        }
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N, const int M_DIM_SIZE, const int BLOCK_PAD_SIZE>
    __device__ void load_tile_t(const int ld, const int dim,
                                const int thread_x, const int thread_y,
                                const int start_block,
                                const int k,
                                const float *__restrict__ global_mem,
                                float *sm) {
#pragma unroll
        for (int j = 0; j < VEC_DIM_M; j++) {
            int dim0 = start_block + thread_y + j * M_DIM_SIZE;
            int dim1 = k + thread_x;
            float val = 0.0f;
            if (dim0 < dim && dim1 < ld) {
                val = __ldg(&global_mem[dim1 + dim0 * ld]);
            }
            sm[thread_x * BLOCK_PAD_SIZE + thread_y + j * M_DIM_SIZE] = val;
        }
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N, const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_PAD_SIZE>
    __device__ void compute_tile(const int thread_x, const int thread_y,
                                 float *a_sm, float *b_sm,
                                 float *c_reg) {
        float a_reg[VEC_DIM_M];
        float b_reg[VEC_DIM_N];
#pragma unroll
        for (int kk = 0; kk < K_DIM_SIZE; kk++) {
#pragma unroll
            for (int j = 0; j < VEC_DIM_M; j++) {
                a_reg[j] = a_sm[kk * BLOCK_PAD_SIZE + thread_x + j * M_DIM_SIZE];
            }
#pragma unroll
            for (int j = 0; j < VEC_DIM_N; j++) {
                b_reg[j] = b_sm[kk * BLOCK_PAD_SIZE + thread_y + j * N_DIM_SIZE];
            }
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N; nn++) {
                    c_reg[nn * VEC_DIM_M + mm] += a_reg[mm] * b_reg[nn];
                }
            }
        }
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N, const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_PAD_SIZE>
    __device__ void store_tile(const int a_ld, const int b_ld, const int c_ld,
                               const int thread_x, const int thread_y,
                               const int start_m, const int start_n,
                               float *__restrict__ c,
                               float *c_reg) {
#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            int m_idx = start_m + thread_x + mm * M_DIM_SIZE;
            if (m_idx >= a_ld) continue;
#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                int n_idx = start_n + thread_y + nn * N_DIM_SIZE;
                if (n_idx >= b_ld) continue;
                c[n_idx * c_ld + m_idx] += c_reg[nn * VEC_DIM_M + mm];
            }
        }
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_nn_pipeline_double_buffer(
        int M, int N, int K,
        int a_ld, int b_ld, int c_ld,
        const float *__restrict__ a,
        const float *__restrict__ b,
        float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE; // 0~15
        const int thread_y = thread_id / N_DIM_SIZE; // 0~15

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE; // 128 * blockIdx.x
        const int start_n = block_y * BLOCK_SIZE;


        __shared__ float a_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];

        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        int flip_flag = 0;
        int k = 0;
        load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                      start_m, k, a, &a_sm[flip_flag][0][0]);
        load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                      start_n, k, b, &b_sm[flip_flag][0][0]);
        __syncthreads();


        for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
            if (k < K) {
                //load 下一块数据到sm;
                load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[!flip_flag][0][0]);
                load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[!flip_flag][0][0]);
            }

            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
            flip_flag ^= 1;
        }
        {
            const int remain_k = K % K_DIM_SIZE;
            for (k = K - remain_k; k < K; k++) {
                load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[flip_flag][0][0]);
                load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[flip_flag][0][0]);
                __syncthreads();
                compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

                __syncthreads();
            }
        }
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_nn_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                                  const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE;
        const int thread_y = thread_id / N_DIM_SIZE;

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE;
        const int start_n = block_y * BLOCK_SIZE;

        __shared__ float a_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
        for (int k = 0; k < K; k += K_DIM_SIZE) {
            load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                          start_m, k, a, &a_sm[0][0]);
            load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                          start_n, k, b, &b_sm[0][0]);
            __syncthreads();
            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }

        __syncthreads();
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_tt_pipeline_double_buffer(
        int M, int N, int K,
        int a_ld, int b_ld, int c_ld,
        const float *__restrict__ a,
        const float *__restrict__ b,
        float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE; // 0~15
        const int thread_y = thread_id / N_DIM_SIZE; // 0~15

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE; // 128 * blockIdx.x
        const int start_n = block_y * BLOCK_SIZE;


        __shared__ float a_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];

        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        int flip_flag = 0;
        int k = 0;
        load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                      start_m, k, a, &a_sm[flip_flag][0][0]);
        load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                      start_n, k, b, &b_sm[flip_flag][0][0]);
        __syncthreads();


        for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
            if (k < K) {
                //load 下一块数据到sm;
                load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[!flip_flag][0][0]);
                load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[!flip_flag][0][0]);
            }

            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
            flip_flag ^= 1;
        }
        {
            const int remain_k = K % K_DIM_SIZE;
            for (k = K - remain_k; k < K; k++) {
                load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[flip_flag][0][0]);
                load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[flip_flag][0][0]);
                __syncthreads();
                compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

                __syncthreads();
            }
        }
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_tt_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                                  const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE;
        const int thread_y = thread_id / N_DIM_SIZE;

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE;
        const int start_n = block_y * BLOCK_SIZE;

        __shared__ float a_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
        for (int k = 0; k < K; k += K_DIM_SIZE) {
            load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                          start_m, k, a, &a_sm[0][0]);
            load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                          start_n, k, b, &b_sm[0][0]);
            __syncthreads();
            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }

        __syncthreads();
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_tn_pipeline_double_buffer(
        int M, int N, int K,
        int a_ld, int b_ld, int c_ld,
        const float *__restrict__ a,
        const float *__restrict__ b,
        float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE; // 0~15
        const int thread_y = thread_id / N_DIM_SIZE; // 0~15

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE; // 128 * blockIdx.x
        const int start_n = block_y * BLOCK_SIZE;


        __shared__ float a_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];

        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        int flip_flag = 0;
        int k = 0;
        load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                      start_m, k, a, &a_sm[flip_flag][0][0]);
        load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                      start_n, k, b, &b_sm[flip_flag][0][0]);
        __syncthreads();


        for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
            if (k < K) {
                //load 下一块数据到sm;
                load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[!flip_flag][0][0]);
                load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[!flip_flag][0][0]);
            }

            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
            flip_flag ^= 1;
        }
        {
            const int remain_k = K % K_DIM_SIZE;
            for (k = K - remain_k; k < K; k++) {
                load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[flip_flag][0][0]);
                load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[flip_flag][0][0]);
                __syncthreads();
                compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

                __syncthreads();
            }
        }
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_tn_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                                  const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE;
        const int thread_y = thread_id / N_DIM_SIZE;

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE;
        const int start_n = block_y * BLOCK_SIZE;

        __shared__ float a_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
        for (int k = 0; k < K; k += K_DIM_SIZE) {
            load_tile_t<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, M, thread_x, thread_y,
                                                                          start_m, k, a, &a_sm[0][0]);
            load_tile_t<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, N, thread_x, thread_y,
                                                                          start_n, k, b, &b_sm[0][0]);
            __syncthreads();
            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }

        __syncthreads();
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_nt_pipeline_double_buffer(
        int M, int N, int K,
        int a_ld, int b_ld, int c_ld,
        const float *__restrict__ a,
        const float *__restrict__ b,
        float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE; // 0~15
        const int thread_y = thread_id / N_DIM_SIZE; // 0~15

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE; // 128 * blockIdx.x
        const int start_n = block_y * BLOCK_SIZE;


        __shared__ float a_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[2][K_DIM_SIZE][BLOCK_PAD_SIZE];

        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        int flip_flag = 0;
        int k = 0;
        load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                      start_m, k, a, &a_sm[flip_flag][0][0]);
        load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                      start_n, k, b, &b_sm[flip_flag][0][0]);
        __syncthreads();


        for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
            if (k < K) {
                //load 下一块数据到sm;
                load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[!flip_flag][0][0]);
                load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[!flip_flag][0][0]);
            }

            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

            __syncthreads();
            flip_flag ^= 1;
        }
        {
            const int remain_k = K % K_DIM_SIZE;
            for (k = K - remain_k; k < K; k++) {
                load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                              start_m, k, a, &a_sm[flip_flag][0][0]);
                load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                              start_n, k, b, &b_sm[flip_flag][0][0]);
                __syncthreads();
                compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0], &c_reg[0]);

                __syncthreads();
            }
        }
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int M_DIM_SIZE, const int N_DIM_SIZE, const int K_DIM_SIZE,
        const int BLOCK_SIZE, const int BLOCK_PAD_SIZE>
    __global__ void sgemm_nt_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                                  const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % N_DIM_SIZE;
        const int thread_y = thread_id / N_DIM_SIZE;

        const int block_x = blockIdx.x;
        const int block_y = blockIdx.y;
        const int start_m = block_x * BLOCK_SIZE;
        const int start_n = block_y * BLOCK_SIZE;

        __shared__ float a_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        __shared__ float b_sm[K_DIM_SIZE][BLOCK_PAD_SIZE];
        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

#pragma unroll
        for (int k = 0; k < K; k += K_DIM_SIZE) {
            load_tile_n<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, K, thread_x, thread_y,
                                                                          start_m, k, a, &a_sm[0][0]);
            load_tile_n<VEC_DIM_M, VEC_DIM_N, N_DIM_SIZE, BLOCK_PAD_SIZE>(b_ld, K, thread_x, thread_y,
                                                                          start_n, k, b, &b_sm[0][0]);
            __syncthreads();
            compute_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(thread_x, thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0]);
            __syncthreads();
        }

        __syncthreads();
        store_tile<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_PAD_SIZE>(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
    }

    template<typename T>
    static void gemm_kernel_cuda(std::vector<std::shared_ptr<tff::core::memory::Tensor> > &src,
                                 std::vector<std::shared_ptr<tff::core::memory::Tensor> > &dst,
                                 std::shared_ptr<core::runtime::LLMWeightMemManager> &mem_buffer_manager_ptr) {
        auto &input_tensor_a = *src.begin();
        auto &input_tensor_b = *src.rbegin();
        auto &output_tensor = *dst.begin();
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


        constexpr int WARP_SIZE = 32;
        constexpr int THREAD_BLOCK_SIZE = 256;
        constexpr int K_DIM_SIZE = 16;
        constexpr int N_DIM_SIZE = THREAD_BLOCK_SIZE / K_DIM_SIZE;
        constexpr int M_DIM_SIZE = N_DIM_SIZE;
        constexpr int VEC_DIM_N = 8;
        constexpr int VEC_DIM_K = 1;
        constexpr int VEC_DIM_M = 8;
        constexpr int BLOCK_SIZE = THREAD_BLOCK_SIZE / K_DIM_SIZE * VEC_DIM_N;
        constexpr int PAD_SIZE = K_DIM_SIZE;
        constexpr int BLOCK_PAD_SIZE = BLOCK_SIZE + PAD_SIZE;

        dim3 grid((S + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (D + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  1);
        dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
        sgemm_nt_func<VEC_DIM_M, VEC_DIM_N, M_DIM_SIZE, N_DIM_SIZE, K_DIM_SIZE, BLOCK_SIZE, BLOCK_PAD_SIZE><<<grid,
                block>>>(S, D, D, S, D, D, static_cast<float *>(input_tensor_a->get_buffer()->ptr()),
                         static_cast<float *>(input_tensor_b->get_buffer()->ptr()), static_cast<float *>(output_tensor->get_buffer()->ptr()));
    }

    template<typename T>
    void tff::kernel::XGemm<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), XGemm<T>::get_op_name().c_str());
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            1, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            2, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(3, para_ptr);

        if (input_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        if (output_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        //
        gemm_kernel_cuda<T>(input_tensors, output_tensors, mem_buffer_manager_ptr);
    }

    template class tff::kernel::XGemm<float>;
    template class tff::kernel::XGemm<double>;
    REGISTER_OP_OBJECT(XGemm, float);

    REGISTER_OP_OBJECT(XGemm, double);
}
