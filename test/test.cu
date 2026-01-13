//
// Created by nkk on 2026/1/12.
//

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>


#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <random>

constexpr int HIDDEN_DIM = 128;
constexpr int BYTES_PER_LOAD = 8; // 128-bit
constexpr int Q_ELEMENTS_PER_LOAD = BYTES_PER_LOAD / sizeof(half2);
constexpr int THREAD_PER_WARP_DIRECTION = HIDDEN_DIM / (sizeof(half2));
constexpr int KV_ELEMENTS_PER_LOAD = HIDDEN_DIM / THREAD_PER_WARP_DIRECTION;
constexpr int WARP_SIZE = 32;
constexpr int THREAD_BLOCK_SIZE = 256;
constexpr int WARP_PER_BLOCK = THREAD_BLOCK_SIZE / THREAD_PER_WARP_DIRECTION;
constexpr int BLOCK_DIM_M = 32; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_M;
constexpr int BLOCK_DIM_N = 16; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_N;
constexpr int VEC_DIM_N = BLOCK_DIM_N / 16;
constexpr int VEC_DIM_M = BLOCK_DIM_M / 16;

template<int B, int M, int S = B>
__device__ int swizzle(const int &offset) {
    const int bit_msk = (1 << B) - 1;
    const int yyy_msk = bit_msk << (M + max(0, S));
    const int zzz_msk = bit_msk << (M - min(0, S));
    const int msk_sft = S;
    if constexpr (S >= 0) {
        return offset ^ ((offset & yyy_msk) >> S);
    } else {
        return offset ^ ((offset & yyy_msk) << -S);
    }
}

__global__ void trans_kernel(int M, int N, float *src, float *dst) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    constexpr int ld_thread_block_n = BLOCK_DIM_N / VEC_DIM_N;
    const int thread_x = thread_id % ld_thread_block_n;
    const int warp_id = thread_id / ld_thread_block_n;

    const int block_x = blockIdx.x;
    const int start_m = block_x * BLOCK_DIM_M;
    const int start_n = blockIdx.y * BLOCK_DIM_N;

    __shared__ float sm[BLOCK_DIM_M * BLOCK_DIM_N];
#pragma unroll
    for (int j = 0; j < VEC_DIM_M; ++j) {
        const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_M / VEC_DIM_M);
        for (int kk = 0; kk < VEC_DIM_N; ++kk) {
            const int dim1 = start_n + thread_x + kk * (BLOCK_DIM_N / VEC_DIM_N);
            int sm_row = warp_id + j * (BLOCK_DIM_M / VEC_DIM_M);
            int sm_col = thread_x + kk * (BLOCK_DIM_N / VEC_DIM_N);
            int offset = sm_row * BLOCK_DIM_N + sm_col;
            //int addr = swizzle<5,0,5>(offset);
            //printf("row: %d, col: %d, swizzle addr: %d\n", sm_row, sm_col, addr);
            sm[offset] = src[dim0_base + dim1 * M];
        }
    }
    __syncthreads();
    for (int j = 0; j < VEC_DIM_M; ++j) {
        int sm_row = warp_id + j * (BLOCK_DIM_M / VEC_DIM_M);
        for (int k = 0; k < VEC_DIM_N; ++k) {
            int sm_col = thread_x + k * (BLOCK_DIM_N / VEC_DIM_N);
            int offset = sm_row * BLOCK_DIM_N + sm_col;
            //int addr = swizzle<5,0,5>(offset);
            dst[(start_m + sm_row) * N + start_n + sm_col] = sm[offset];
        }
    }
}

template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}

int main7678(int argc, char **argv) {
    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);
    int dim = 32;
    int m = dim;
    int n = dim;

    std::vector<float> a_mat;
    a_mat.resize(m * n);
    std::vector<float> b_mat;
    b_mat.resize(n * m);

    PopulateVector<float>(a_mat, mt, dist);
    //PopulateVector<float>(b_mat, mt, dist);


    float *a_gpu = nullptr;
    cudaMalloc((void **) &a_gpu, m * n * sizeof(float));
    cudaMemcpy(a_gpu, a_mat.data(), m * n * sizeof(float), cudaMemcpyHostToDevice);
    float *b_gpu = nullptr;
    cudaMalloc((void **) &b_gpu, m * n * sizeof(float));


    dim3 grid((m + BLOCK_DIM_M - 1) / BLOCK_DIM_M, (n + BLOCK_DIM_N - 1) / BLOCK_DIM_N);
    dim3 block(32, 8);
    printf("grid x: %d, grid y: %d, grid z: %d \n", grid.x, grid.y, grid.z);
    printf("block x: %d, block y: %d\n", block.x, block.y);
    trans_kernel<<<grid, block>>>(m, n, a_gpu, b_gpu);
    cudaDeviceSynchronize();


    cudaMemcpy(b_mat.data(), b_gpu, m * n * sizeof(float), cudaMemcpyDeviceToHost);


    for (int j = 0; j < m; ++j) {
        for (int k = 0; k < n; ++k) {
            float delta = a_mat[j * n + k] - b_mat[j + k * m];
            if (delta != 0.0) {
                printf("error: %f, m: %d, n: %d\n", delta, j, k);
                return 0;
            }
        }
    }
    printf("success\n");

    cudaFree(a_gpu);
    a_gpu = nullptr;
    cudaFree(b_gpu);
    b_gpu = nullptr;
    return 0;
}
