//
// Created by nkk on 2025/11/23.
//
#include <vector>
#include <random>
#include "cublas_v2.h"
#include "mma.h"
#include <cstdint>
#include <cstring>

// #include "cutlass/gemm/device/gemm.h"
// #include "cutlass/gemm/device/gemm_universal_adapter.h"
// #include "cutlass/numeric_types.h"
// #include "cutlass/layout/matrix.h"
// #include <cuda_runtime.h>
// #include <iostream>


inline float to_tf32(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    // TF32: keep sign (1b) + exponent (8b) + top 10b of mantissa → total 19 bits
    // Clear the lower 13 bits of mantissa (bits 0~12)
    bits &= 0xFFFFE000U; // 1111 1111 1111 1111 1110 0000 0000 0000

    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

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
    load_tile_n(a_ld, K, thread_x, thread_y,
              start_m, k, a,&a_sm[flip_flag][0][0]);
    load_tile_t(b_ld, N, thread_x, thread_y,
              start_n, k, b,&b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_n(a_ld, K, thread_x, thread_y,
              start_m, k, a,&a_sm[!flip_flag][0][0]);
            load_tile_t(b_ld, N, thread_x, thread_y,
                      start_n, k, b,&b_sm[!flip_flag][0][0]);
        }

        compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % K_DIM_SIZE;
        for (k = K - remain_k; k < K; k++) {
            load_tile_n(a_ld, K, thread_x, thread_y,
             start_m, k, a,&a_sm[flip_flag][0][0]);
            load_tile_t(b_ld, N, thread_x, thread_y,
                      start_n, k, b,&b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

            __syncthreads();
        }
    }
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
               start_m, start_n, c,
               &c_reg[0]);
}
__global__ void sgemm_nn_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                         float *a, float *b, float *c) {
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
        load_tile_n(a_ld, K, thread_x, thread_y,
              start_m, k, a,&a_sm[0][0]);
        load_tile_t(b_ld, N, thread_x, thread_y,
                  start_n, k, b,&b_sm[0][0]);
        __syncthreads();
        compute_tile(thread_x, thread_y,&a_sm[0][0], &b_sm[0][0],&c_reg[0]);
        __syncthreads();
    }

    __syncthreads();
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
}
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
    load_tile_t(a_ld, M, thread_x, thread_y,
              start_m, k, a,&a_sm[flip_flag][0][0]);
    load_tile_n(b_ld, K, thread_x, thread_y,
              start_n, k, b,&b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_t(a_ld, M, thread_x, thread_y,
              start_m, k, a,&a_sm[!flip_flag][0][0]);
            load_tile_n(b_ld, K, thread_x, thread_y,
                      start_n, k, b,&b_sm[!flip_flag][0][0]);
        }

        compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % K_DIM_SIZE;
        for (k = K - remain_k; k < K; k++) {
            load_tile_t(a_ld, M, thread_x, thread_y,
             start_m, k, a,&a_sm[flip_flag][0][0]);
            load_tile_n(b_ld, K, thread_x, thread_y,
                      start_n, k, b,&b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

            __syncthreads();
        }
    }
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
               start_m, start_n, c,
               &c_reg[0]);
}
__global__ void sgemm_tt_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                         float *a, float *b, float *c) {
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
        load_tile_t(a_ld, M, thread_x, thread_y,
              start_m, k, a,&a_sm[0][0]);
        load_tile_n(b_ld, K, thread_x, thread_y,
                  start_n, k, b,&b_sm[0][0]);
        __syncthreads();
        compute_tile(thread_x, thread_y,&a_sm[0][0], &b_sm[0][0],&c_reg[0]);
        __syncthreads();
    }

    __syncthreads();
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
}
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
    load_tile_t(a_ld, M, thread_x, thread_y,
              start_m, k, a,&a_sm[flip_flag][0][0]);
    load_tile_t(b_ld, N, thread_x, thread_y,
              start_n, k, b,&b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_t(a_ld, M, thread_x, thread_y,
              start_m, k, a,&a_sm[!flip_flag][0][0]);
            load_tile_t(b_ld, N, thread_x, thread_y,
                      start_n, k, b,&b_sm[!flip_flag][0][0]);
        }

        compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % K_DIM_SIZE;
        for (k = K - remain_k; k < K; k++) {
            load_tile_t(a_ld, M, thread_x, thread_y,
             start_m, k, a,&a_sm[flip_flag][0][0]);
            load_tile_t(b_ld, N, thread_x, thread_y,
                      start_n, k, b,&b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

            __syncthreads();
        }
    }
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
               start_m, start_n, c,
               &c_reg[0]);
}
__global__ void sgemm_tn_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                         float *a, float *b, float *c) {
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
        load_tile_t(a_ld, M, thread_x, thread_y,
              start_m, k, a,&a_sm[0][0]);
        load_tile_t(b_ld, N, thread_x, thread_y,
                  start_n, k, b,&b_sm[0][0]);
        __syncthreads();
        compute_tile(thread_x, thread_y,&a_sm[0][0], &b_sm[0][0],&c_reg[0]);
        __syncthreads();
    }

    __syncthreads();
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
}
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
    load_tile_n(a_ld, K, thread_x, thread_y,
              start_m, k, a,&a_sm[flip_flag][0][0]);
    load_tile_n(b_ld, K, thread_x, thread_y,
              start_n, k, b,&b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile_n(a_ld, K, thread_x, thread_y,
              start_m, k, a,&a_sm[!flip_flag][0][0]);
            load_tile_n(b_ld, K, thread_x, thread_y,
                      start_n, k, b,&b_sm[!flip_flag][0][0]);
        }

        compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    {
        const int remain_k = K % K_DIM_SIZE;
        for (k = K - remain_k; k < K; k++) {
            load_tile_n(a_ld, K, thread_x, thread_y,
             start_m, k, a,&a_sm[flip_flag][0][0]);
            load_tile_n(b_ld, K, thread_x, thread_y,
                      start_n, k, b,&b_sm[flip_flag][0][0]);
            __syncthreads();
            compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

            __syncthreads();
        }
    }
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
               start_m, start_n, c,
               &c_reg[0]);
}
__global__ void sgemm_nt_func(int M, int N, int K, int a_ld, int b_ld, int c_ld,
                         float *a, float *b, float *c) {
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
        load_tile_n(a_ld, K, thread_x, thread_y,
              start_m, k, a,&a_sm[0][0]);
        load_tile_n(b_ld, K, thread_x, thread_y,
                  start_n, k, b,&b_sm[0][0]);
        __syncthreads();
        compute_tile(thread_x, thread_y,&a_sm[0][0], &b_sm[0][0],&c_reg[0]);
        __syncthreads();
    }

    __syncthreads();
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
                   start_m, start_n, c,
                   &c_reg[0]);
}

template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}

void gemm_nn_cpu(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[p * m + i] * b[p + j * k];
            }
            c[j * m + i] = sum; // C(i, j) in column-major
        }
    }
}

void gemm_nt_cpu(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[p * m + i] * b[p * n + j];
            }
            c[j * m + i] = sum; // C(i, j) in column-major
        }
    }
}

void gemm_tt_cpu(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[p + i * k] * b[p * n + j];
            }
            c[j * m + i] = sum; // C(i, j) in column-major
        }
    }
}

void gemm_tn_cpu(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += a[p + i * k] * b[p + j * k];
            }
            c[j * m + i] = sum; // C(i, j) in column-major
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
void sgemm_nn_func(int m, int n, int k,
           std::vector<float> &a_mat,
           std::vector<float> &b_mat,
           std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nn_func<<<grid, block>>>(m, n, k, m, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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

    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 func Performance: %.2f GFLOPS/s\n \n", gflops_per_sec);

#ifdef _DEBUG
    gemm_nn_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
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
#endif
}

void sgemm_nn_double_buffer(int m, int n, int k,
                         std::vector<float> &a_mat,
                         std::vector<float> &b_mat,
                         std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nn_pipeline_double_buffer<<<grid, block>>>(m, n, k, m, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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

    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_nn_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
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
#endif
}
void sgemm_tt_func(int m, int n, int k,
           std::vector<float> &a_mat,
           std::vector<float> &b_mat,
           std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tt_func<<<grid, block>>>(m, n, k, k, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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

    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 tt func Performance: %.2f GFLOPS/s\n \n", gflops_per_sec);

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
#endif
}

void sgemm_tt_double_buffer(int m, int n, int k,
                         std::vector<float> &a_mat,
                         std::vector<float> &b_mat,
                         std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tt_pipeline_double_buffer<<<grid, block>>>(m, n, k, k, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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
#endif
}
void sgemm_tn_func(int m, int n, int k,
           std::vector<float> &a_mat,
           std::vector<float> &b_mat,
           std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tn_func<<<grid, block>>>(m, n, k, k, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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
#endif
}

void sgemm_tn_double_buffer(int m, int n, int k,
                         std::vector<float> &a_mat,
                         std::vector<float> &b_mat,
                         std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_tn_pipeline_double_buffer<<<grid, block>>>(m, n, k, k, k, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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
#endif
}
void sgemm_nt_func(int m, int n, int k,
           std::vector<float> &a_mat,
           std::vector<float> &b_mat,
           std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nt_func<<<grid, block>>>(m, n, k, m, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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

    printf("Matrix size: %d x %d x %d\n", m, n, k);
    printf("Kernel time: %.4f ms\n", milliseconds);
    printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
    printf("手写 func nt Performance: %.2f GFLOPS/s\n", gflops_per_sec);

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
#endif
}

void sgemm_nt_double_buffer(int m, int n, int k,
                         std::vector<float> &a_mat,
                         std::vector<float> &b_mat,
                         std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    dim3 grid((m + BLOCK_SIZE - 1) / BLOCK_SIZE,
              (n + BLOCK_SIZE - 1) / BLOCK_SIZE,
              1);
    dim3 block(THREAD_BLOCK_SIZE / K_DIM_SIZE, K_DIM_SIZE, 1);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));
    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    sgemm_nt_pipeline_double_buffer<<<grid, block>>>(m, n, k, m, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat_gpu_result.data(), c_mat_gpu, c_mat.size() * sizeof(float),
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
#endif
}
int main3(int argc, char *argv) {
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, 0);
    printf("device prop sharedMemPerBlock:%d \n", device_prop.sharedMemPerBlock);
    printf("device prop regsPerBlock: %d\n", device_prop.regsPerBlock);
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
#ifdef _DEBUG
    int m = 2048;
    int n = 2048;
    int k = 2048;

    std::vector<float> a_mat;
    a_mat.resize(m * k);
    std::vector<float> b_mat;
    b_mat.resize(n * k);
    std::vector<float> c_mat;
    c_mat.resize(m * n);
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    PopulateVector<float>(a_mat, mt, dist);
    PopulateVector<float>(b_mat, mt, dist);

    // sgemm_nn_func(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_nn_double_buffer(m, n, k, a_mat, b_mat, c_mat);
    //
    // sgemm_nt_func(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_nt_double_buffer(m, n, k, a_mat, b_mat, c_mat);
    //
    // sgemm_tn_func(m, n, k, a_mat, b_mat, c_mat);
    // sgemm_tn_double_buffer(m, n, k,a_mat, b_mat, c_mat);

    sgemm_tn_func(m, n, k, a_mat, b_mat, c_mat);
    sgemm_tn_double_buffer(m, n, k,a_mat, b_mat, c_mat);
    //
    //sgemm_wmma(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //sgemm_wmma_sm(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //
    //cutlass_sgemm_tf32(m, n, k, m, n, m, a_mat, b_mat, c_mat);
#else
    int n = 128;
    int m = 128;
    int k = 128;
    for (int j = 1; j < 18; j++) {
        n = m = k += 2 * 128;
        printf("*******************************\n");
        printf("m: %d,n: %d, k: %d \n",m, n, k);
        std::vector<float> a_mat;
        a_mat.resize(m * k);
        std::vector<float> b_mat;
        b_mat.resize(n * k);
        std::vector<float> c_mat;
        c_mat.resize(m * n);
        std::vector<float> c_mat_gpu_result;
        c_mat_gpu_result.resize(m * n);

        PopulateVector<float>(a_mat, mt, dist);
        PopulateVector<float>(b_mat, mt, dist);

        //sgemm_nn(m, n, k, m, n, m, a_mat, b_mat, c_mat);
        //sgemm_nn_func(m, n, k, m, n, m, a_mat, b_mat, c_mat);
        //sgemm_nn_double_buffer(m, n, k, m, n, m, a_mat, b_mat, c_mat);
    }

    //
    //sgemm_wmma(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //sgemm_wmma_sm(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //
    //cutlass_sgemm_tf32(m, n, k, m, n, m, a_mat, b_mat, c_mat);
#endif


    return 0;
}
