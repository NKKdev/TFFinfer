//
// Created by nkk on 2025/11/23.
//
#include <vector>
#include <random>
#include "cublas_v2.h"
#include "mma.h"
#include <cstdint>
#include <cstring>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include <cuda_runtime.h>
#include <iostream>


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

__device__ void load_tile(const int a_ld, const int b_ld, const int K,
                          const int thread_x, const int thread_y,
                          const int start_m, const int start_n,
                          const int k,
                          const float *__restrict__ a, const float *__restrict__ b,
                          float *a_sm, float *b_sm) {
#pragma unroll
    for (int j = 0; j < VEC_DIM_M; j++) {
        int m_idx = start_m + thread_x + j * M_DIM_SIZE;
        int k_idx = k + thread_y;
        float a_val = 0.0f;
        if (k_idx < K && m_idx < a_ld) {
            a_val = __ldg(&a[k_idx * a_ld + m_idx]);
        }
        a_sm[thread_y * BLOCK_PAD_SIZE + thread_x + j * M_DIM_SIZE] = a_val;

        int n_idx = start_n + thread_x + j * N_DIM_SIZE;
        float b_val = 0.0f;
        if (k_idx < K && n_idx < b_ld) {
            b_val = __ldg(&b[k_idx * b_ld + n_idx]);
        }
        b_sm[thread_y * BLOCK_PAD_SIZE + thread_x + j * N_DIM_SIZE] = b_val;
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
    load_tile(a_ld, b_ld, K, thread_x, thread_y,
              start_m, start_n, k, a, b, &a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0]);
    __syncthreads();


    for (k = K_DIM_SIZE; k <= K; k += K_DIM_SIZE) {
        if (k < K) {
            //load 下一块数据到sm;
            load_tile(a_ld, b_ld, K, thread_x, thread_y,
                      start_m, start_n, k, a, b, &a_sm[!flip_flag][0][0], &b_sm[!flip_flag][0][0]);
        }

        compute_tile(thread_x, thread_y,&a_sm[flip_flag][0][0], &b_sm[flip_flag][0][0],&c_reg[0]);

        __syncthreads();
        flip_flag ^= 1;
    }
    store_tile(a_ld, b_ld, c_ld, thread_x, thread_y,
               start_m, start_n, c,
               &c_reg[0]);
}

__global__ void sgemm_nt(int M, int N, int K, int a_ld, int b_ld, int c_ld,
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
#pragma unroll
        for (int j = 0; j < VEC_DIM_M; j++) {
            int m_idx = start_m + thread_x + j * M_DIM_SIZE;
            int k_idx = k + thread_y;
            float a_val = 0.0f;
            if (k_idx < K && m_idx < M) {
                a_val = __ldg(&a[k_idx * a_ld + m_idx]);
            }
            a_sm[thread_y][thread_x + j * M_DIM_SIZE] = a_val;
            //
            int n_idx = start_n + thread_x + j * N_DIM_SIZE;
            float b_val = 0.0f;
            if (k_idx < K && n_idx < N) {
                b_val = __ldg(&b[k_idx * b_ld + n_idx]);
            }
            b_sm[thread_y][thread_x + j * N_DIM_SIZE] = b_val;
        }
        __syncthreads();
        float a_reg[VEC_DIM_M] = {0};
        float b_reg[VEC_DIM_N] = {0};
#pragma unroll
        for (int kk = 0; kk < K_DIM_SIZE; kk++) {
#pragma unroll
            for (int j = 0; j < VEC_DIM_M; j++) {
                a_reg[j] = a_sm[kk][thread_x + j * M_DIM_SIZE];
                b_reg[j] = b_sm[kk][thread_y + j * N_DIM_SIZE];
            }
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N; nn++) {
                    c_reg[nn * VEC_DIM_M + mm] += a_reg[mm] * b_reg[nn];
                }
            }
        }
        __syncthreads();
    }

    __syncthreads();
#pragma unroll
    for (int mm = 0; mm < VEC_DIM_M; mm++) {
        int m_idx = start_m + thread_x + mm * M_DIM_SIZE;
        if (m_idx >= M) {
            continue;
        }
#pragma unroll
        for (int nn = 0; nn < VEC_DIM_N; nn++) {
            int n_idx = start_n + thread_y + nn * N_DIM_SIZE;
            if (n_idx >= N) {
                continue;
            }
            c[n_idx * c_ld + m_idx] += c_reg[nn * VEC_DIM_M + mm];
        }
    }
}

constexpr int M_DIM = 16;
constexpr int N_DIM = 16;
constexpr int K_DIM = 8;

constexpr int WARPS_PER_BLOCK_M = 2; // 例如 2 warps in M direction
constexpr int WARPS_PER_BLOCK_N = 2; // 4*4=16 warps, 16*32=512 threads
constexpr int WMMA_THREAD_BLOCK_SIZE = WARPS_PER_BLOCK_M * WARPS_PER_BLOCK_N * WARP_SIZE;
constexpr int BLOCK_M = WARPS_PER_BLOCK_M * M_DIM;
constexpr int BLOCK_N = WARPS_PER_BLOCK_N * N_DIM;
constexpr int SMEM_PAD = 32;
constexpr int K_PAD_M = K_DIM + SMEM_PAD;
constexpr int K_PAD_N = K_DIM + SMEM_PAD;


__global__ void sgemm_nt_wmma_sm(int M, int N, int K,

                                 int a_ld, int b_ld, int c_ld,

                                 const float *__restrict__ a,

                                 const float *__restrict__ b,

                                 float *__restrict__ c) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int warp_m = warp_id % WARPS_PER_BLOCK_M; // warp 在 M 方向索引
    const int warp_n = warp_id / WARPS_PER_BLOCK_M; // warp 在 N 方向索引

    const int block_m = blockIdx.x * WARPS_PER_BLOCK_M * M_DIM; //64
    const int block_n = blockIdx.y * WARPS_PER_BLOCK_N * N_DIM; //64

    const int m_start = block_m + warp_m * M_DIM;
    const int n_start = block_n + warp_n * N_DIM;

    __shared__ float sa[BLOCK_M][K_PAD_M]; // [32][16+pad]
    __shared__ float sb[BLOCK_N][K_PAD_N]; // [32][16+pad]


    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M_DIM, N_DIM, K_DIM, nvcuda::wmma::precision::tf32,
        nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M_DIM, N_DIM, K_DIM, nvcuda::wmma::precision::tf32,
        nvcuda::wmma::col_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M_DIM, N_DIM, K_DIM, float> frag_c;


    nvcuda::wmma::fill_fragment(frag_c, 0.0f);

    const int tid = threadIdx.x;
    const int threads_per_block = blockDim.x;
#pragma unroll
    for (int k = 0; k < K; k += K_DIM) {
#pragma unroll
        for (int i = tid; i < BLOCK_M * K_DIM; i += threads_per_block) {
            const int row = i / K_DIM; // [0, BLOCK_M)
            const int col = i % K_DIM; // [0, K_DIM)
            const int global_k = k + col;

            float val = 0.0f;
            if (block_m + row < M && global_k < K) {
                val = a[(block_m + row) * a_ld + global_k];
            }
            sa[row][col] = val;
        }
#pragma unroll
        for (int i = tid; i < BLOCK_N * K_DIM; i += threads_per_block) {
            const int irow = i / K_DIM; // [0, BLOCK_N)
            const int icol = i % K_DIM; // [0, K_DIM)
            const int global_k = k + icol;
            float val = 0.0f;
            if (block_n + irow < N && global_k < K) {
                val = b[(block_n + irow) * b_ld + global_k];
            }
            sb[irow][icol] = val;
        }
        __syncthreads();


        nvcuda::wmma::load_matrix_sync(
            frag_a,
            &sa[warp_m * M_DIM][0],
            K_PAD_M
        );


        nvcuda::wmma::load_matrix_sync(
            frag_b,
            &sb[warp_n * N_DIM][0],
            K_PAD_N

        );
        __syncthreads();
        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        __syncthreads();
    }

    if (m_start < M && n_start < N) {
        float *c_ptr = &c[n_start * c_ld + m_start];
        nvcuda::wmma::store_matrix_sync(c_ptr, frag_c, c_ld, nvcuda::wmma::mem_col_major);
    }
}

__global__ void sgemm_nt_wmma(int M, int N, int K,

                              int a_ld, int b_ld, int c_ld,

                              const float *__restrict__ a,

                              const float *__restrict__ b,

                              float *__restrict__ c) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int warp_m = warp_id % WARPS_PER_BLOCK_M; // warp 在 M 方向索引
    const int warp_n = warp_id / WARPS_PER_BLOCK_M; // warp 在 N 方向索引

    const int block_m = blockIdx.x * WARPS_PER_BLOCK_M * M_DIM; //64
    const int block_n = blockIdx.y * WARPS_PER_BLOCK_N * N_DIM; //64

    const int m_start = block_m + warp_m * M_DIM;
    const int n_start = block_n + warp_n * N_DIM;

    __shared__ float sa[BLOCK_M][K_DIM + SMEM_PAD]; // [32][16+pad]
    __shared__ float sb[BLOCK_N][K_DIM + SMEM_PAD]; // [32][16+pad]

    if (m_start >= M || n_start >= N) return;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M_DIM, N_DIM, K_DIM, nvcuda::wmma::precision::tf32,
        nvcuda::wmma::row_major> frag_a;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M_DIM, N_DIM, K_DIM, nvcuda::wmma::precision::tf32,
        nvcuda::wmma::col_major> frag_b;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M_DIM, N_DIM, K_DIM, float> frag_c;


    nvcuda::wmma::fill_fragment(frag_c, 0.0f);


    for (int k = 0; k < K; k += K_DIM) {
        const float *a_ptr = &a[m_start * a_ld + k]; // a_ld = K
        nvcuda::wmma::load_matrix_sync(frag_a, a_ptr, a_ld);


        const float *b_ptr = &b[n_start * b_ld + k];
        nvcuda::wmma::load_matrix_sync(frag_b, b_ptr, b_ld);


        nvcuda::wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    float *c_ptr = &c[n_start * c_ld + m_start];
    nvcuda::wmma::store_matrix_sync(c_ptr, frag_c, c_ld, nvcuda::wmma::mem_col_major);
}

template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}

void gemm_cpu(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++) {
        // row of C (and row of A)
        for (int j = 0; j < n; j++) {
            // column of C (and row of B)
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                // common dimension
                // A(i, p) = a[p * m + i]
                // B(j, p) = b[p * n + j]  --> because B^T(p, j) = B(j, p)
                sum += a[p * m + i] * b[p * n + j];
            }
            c[j * m + i] = sum; // C(i, j) in column-major
        }
    }
}

void gemm_cpu_col_major(int m, int n, int k, const float *A, const float *B, float *C) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += to_tf32(A[p * m + i]) * to_tf32(B[j * k + p]); // A(i,p) * B(p,j)
            }
            C[j * m + i] = sum;
        }
    }
}

//
void gemm_cpu_wmma(int m, int n, int k, float *a, float *b, float *c) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                float a_val = to_tf32(a[i * k + p]); // A input as TF32
                float b_val = to_tf32(b[j * k + p]); // B input as TF32
                sum += a_val * b_val; // Multiply in FP32, accumulate in FP32
            }
            c[j * m + i] = sum; // column-major
        }
    }
}

void cutlass_sgemm_tf32(int m, int n, int k,
                        int a_ld, int b_ld, int c_ld,
                        std::vector<float> &a_mat,
                        std::vector<float> &b_mat,
                        std::vector<float> &c_mat) {
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::tfloat32_t, // ElementA (TF32)
        cutlass::layout::ColumnMajor, // LayoutA
        cutlass::tfloat32_t, // ElementB (TF32)
        cutlass::layout::ColumnMajor, // LayoutB
        float, // ElementC (FP32 accumulator)
        cutlass::layout::ColumnMajor // LayoutC
    >;

    // (M, N), row-major
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);

    float *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, a_mat.size() * sizeof(float));
    cudaMemcpy(a_mat_gpu, a_mat.data(), a_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, b_mat.size() * sizeof(float));
    cudaMemcpy(b_mat_gpu, b_mat.data(), b_mat.size() * sizeof(float), cudaMemcpyHostToDevice);
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, c_mat.size() * sizeof(float));
    cudaMemset(c_mat_gpu, 0.0f, c_mat.size() * sizeof(float));


    const auto *ptr_A = reinterpret_cast<const cutlass::tfloat32_t *>(a_mat_gpu);
    const auto *ptr_B = reinterpret_cast<const cutlass::tfloat32_t *>(b_mat_gpu);
    float *ptr_C = c_mat_gpu;

    typename Gemm::Arguments args(
        {m, n, k}, // GemmShape
        {ptr_A, k}, // TensorRef A (data, lda)
        {ptr_B, n}, // TensorRef B (data, ldb)
        {ptr_C, n}, // TensorRef C (data, ldc)
        {ptr_C, n}, // TensorRef D (output)
        {1.0f, 0.0f} // alpha, beta
    );

    Gemm gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(args);
    void *workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cutlass::Status status = gemm_op(args, workspace);
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
    printf("cutlass Performance: %.2f GFLOPS/s\n", gflops_per_sec);

    if (workspace) cudaFree(workspace);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS GEMM failed!" << std::endl;
        return;
    }

#ifdef _DEBUG
    gemm_cpu_col_major(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.01f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm_wmma sucess !!: \n");
#endif
}

void sgemm_wmma_sm(int m, int n, int k,

                   int a_ld, int b_ld, int c_ld,
                   std::vector<float> &a_mat,
                   std::vector<float> &b_mat,
                   std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);


    dim3 grid((m + WARPS_PER_BLOCK_M * M_DIM - 1) / (WARPS_PER_BLOCK_M * M_DIM),

              (n + WARPS_PER_BLOCK_N * N_DIM - 1) / (WARPS_PER_BLOCK_N * N_DIM));
    dim3 block(WMMA_THREAD_BLOCK_SIZE, 1, 1); // 256
    printf("grid x: %d, y: %d \n", grid.x, grid.y);
    printf("block x: %d, y: %d \n", block.x, block.y);
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
    sgemm_nt_wmma_sm<<<grid, block>>>(m, n, k, a_ld, b_ld, c_ld, a_mat_gpu, b_mat_gpu, c_mat_gpu);
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
    printf("wmma with sm Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_cpu_wmma(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    // //cpu:
    // printf("cpu c: \n");
    // for (int i = 0; i < m; i++) {
    //     // row of C (and row of A)
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ",c_mat[j * m + i]);
    //     }
    //     printf("\n");
    // }
    // //gpu:
    // printf("gpu c: \n");
    // for (int i = 0; i < m; i++) {
    //     // row of C (and row of A)
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ",c_mat_gpu_result[j * m + i]);
    //     }
    //     printf("\n");
    // }

    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.01f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm_wmma sucess !!: \n");
#endif
}

void sgemm_wmma(int m, int n, int k,

                int a_ld, int b_ld, int c_ld,
                std::vector<float> &a_mat,
                std::vector<float> &b_mat,
                std::vector<float> &c_mat) {
    std::vector<float> c_mat_gpu_result;
    c_mat_gpu_result.resize(m * n);


    dim3 grid((m + WARPS_PER_BLOCK_M * M_DIM - 1) / (WARPS_PER_BLOCK_M * M_DIM),

              (n + WARPS_PER_BLOCK_N * N_DIM - 1) / (WARPS_PER_BLOCK_N * N_DIM));
    dim3 block(WMMA_THREAD_BLOCK_SIZE, 1, 1); // 256
    printf("grid x: %d, y: %d \n", grid.x, grid.y);
    printf("block x: %d, y: %d \n", block.x, block.y);
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
    sgemm_nt_wmma<<<grid, block>>>(m, n, k, a_ld, b_ld, c_ld, a_mat_gpu, b_mat_gpu, c_mat_gpu);
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
    printf("wmma no sm Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_cpu_wmma(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
    // //cpu:
    // printf("cpu c: \n");
    // for (int i = 0; i < m; i++) {
    //     // row of C (and row of A)
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ",c_mat[j * m + i]);
    //     }
    //     printf("\n");
    // }
    // //gpu:
    // printf("gpu c: \n");
    // for (int i = 0; i < m; i++) {
    //     // row of C (and row of A)
    //     for (int j = 0; j < n; j++) {
    //         printf("%lf ",c_mat_gpu_result[j * m + i]);
    //     }
    //     printf("\n");
    // }

    printf("result c: \n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_gpu_result[j * m + i] - c_mat[j * m + i];

            if (fabs(delta) > 0.01f) {
                printf("error m: %d, n %d, delta: %lf, gpu: %lf, cpu: %lf\n", i, j, delta, c_mat_gpu_result[j * m + i],
                       c_mat[j * m + i]);
                return;
            }
            //printf("%lf ", delta);
        }
        //printf("\n");
    }
    printf("sgemm_wmma sucess !!: \n");
#endif
}

void sgemm(int m, int n, int k,

           int a_ld, int b_ld, int c_ld,
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
    sgemm_nt<<<grid, block>>>(m, n, k, m, n, m, a_mat_gpu, b_mat_gpu, c_mat_gpu);
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
    printf("手写 Performance: %.2f GFLOPS/s\n \n", gflops_per_sec);

#ifdef _DEBUG
    gemm_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
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

void sgemm_double_buffer(int m, int n, int k,
                         int a_ld, int b_ld, int c_ld,
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
    printf("手写 double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);

#ifdef _DEBUG
    gemm_cpu(m, n, k, a_mat.data(), b_mat.data(), c_mat.data());
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

int main(int argc, char *argv) {
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, 0);
    printf("device prop sharedMemPerBlock:%d \n", device_prop.sharedMemPerBlock);
    printf("device prop regsPerBlock: %d\n", device_prop.regsPerBlock);
    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
#ifdef _DEBUG
    int n = 128;
    int m = 128;
    int k = 128;

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

    sgemm(m, n, k, m, n, m, a_mat, b_mat, c_mat);
    sgemm_double_buffer(m, n, k, m, n, m, a_mat, b_mat, c_mat);
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

        sgemm(m, n, k, m, n, m, a_mat, b_mat, c_mat);
        sgemm_double_buffer(m, n, k, m, n, m, a_mat, b_mat, c_mat);
    }

    //
    //sgemm_wmma(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //sgemm_wmma_sm(m,n,k, m, n,m,a_mat,b_mat,c_mat);
    //
    //cutlass_sgemm_tf32(m, n, k, m, n, m, a_mat, b_mat, c_mat);
#endif


    return 0;
}
