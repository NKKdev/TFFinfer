//
// Created by nkk on 2025/12/1.
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
const int BLOCK_M_DIM = 8;
const int BLOCK_N_DIM = 16;
const int BLOCK_K_DIM = BLOCK_N_DIM;
const int PAD_SIZE = 1;
const int BLOCK_PAD_SIZE = BLOCK_K_DIM + PAD_SIZE;
constexpr int VEC_DIM_M = 1;
constexpr int VEC_DIM_N = VEC_DIM_M * BLOCK_N_DIM / BLOCK_M_DIM;
constexpr int VEC_DIM_K = 4;
constexpr int VEC_DOT_PRODUCT = 8;


template<const int QUANT_BLOCK_SIZE>
__device__ void load_tile_a(const int ld, const int dim,
                          const int thread_x, const int thread_y,
                          const int start_block,
                          const int k,
                          const tff::core::quant::Q_8_0 *__restrict__ global_mem,
                          tff::core::quant::Q_8_0 *sm) {
    for (int j = 0; j < VEC_DIM_M;++j) {
        int dim0 = start_block + thread_y + j * BLOCK_M_DIM / VEC_DIM_M;
        int dim1 = k + thread_x;
        tff::core::quant::Q_8_0 val;
        val.d = 0;
        val.qs[QUANT_BLOCK_SIZE] = {0};
        if (dim1 < ld && dim0 < dim) {
            val = global_mem[dim0 * ld + dim1];
        }
        sm[(thread_y + j * BLOCK_M_DIM / VEC_DIM_M) * BLOCK_PAD_SIZE + thread_x] = val;
    }

}
template<const int QUANT_BLOCK_SIZE>
__device__ void load_tile_b(const int ld, const int dim,
                          const int thread_x, const int thread_y,
                          const int start_block,
                          const int k,
                          const tff::core::quant::Q_8_0 *__restrict__ global_mem,
                          tff::core::quant::Q_8_0 *sm) {
    for (int j = 0; j < VEC_DIM_N;++j) {
        int dim0 = start_block + thread_y + j * BLOCK_N_DIM / VEC_DIM_N;
        int dim1 = k + thread_x;
        tff::core::quant::Q_8_0 val;
        val.d = 0;
        val.qs[QUANT_BLOCK_SIZE] = {0};
        if (dim1 < ld && dim0 < dim) {
            val = global_mem[dim0 * ld + dim1];
        }
        sm[(thread_y + j * BLOCK_N_DIM / VEC_DIM_N) * BLOCK_PAD_SIZE + thread_x] = val;
    }

}
__device__ int vec_dot_product(int a, int b, int c_sum) {
    return __dp4a(a, b, c_sum);
}

template<const int QUANT_BLOCK_SIZE>
__device__ void compute_tile(const int thread_x, const int thread_y,tff::core::quant::Q_8_0 *a_sm, tff::core::quant::Q_8_0 *b_sm,
                             float *c_reg) {
    float sum = 0;
#pragma unroll
    for (int kk = 0; kk < BLOCK_K_DIM; kk++) {
        tff::core::quant::Q_8_0 a_reg = a_sm[thread_y * BLOCK_PAD_SIZE + kk];
        tff::core::quant::Q_8_0 b_reg = b_sm[thread_x * BLOCK_PAD_SIZE + kk];
        int32_t *a_k_data = reinterpret_cast<int32_t *>(&a_reg.qs[0]);
        float d_a = __half2float(a_reg.d);
        int32_t *b_k_data = reinterpret_cast<int32_t *>(&b_reg.qs[0]);
        float d_b = __half2float(b_reg.d);
        float block_sum = 0;
#pragma unroll
        for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q++) {
            block_sum = vec_dot_product(a_k_data[kk_q], b_k_data[kk_q], block_sum);
        }
        // if (kk == 0 && thread_y == 0 && thread_x == 1 && blockIdx.y == 0 && blockIdx.x == 0) {
        //     printf("sum: %f \n", block_sum);
        //     for (int kk_q = 0; kk_q < QUANT_BLOCK_SIZE; kk_q++) {
        //         printf("a_reg.qs[%d]:%d, b_reg.qs[%d]:%d, d_a: %f, d_b: %f \n", kk_q,a_reg.qs[kk_q], kk_q,b_reg.qs[kk_q],
        //             d_a, d_b);
        //     }
        // }

        sum += block_sum * d_a * d_b;
        // if (thread_y == 0 && thread_x == 1 && blockIdx.y == 0 && blockIdx.x == 0) {
        //     printf("c_sum: %f \n", sum);
        // }
    }

    c_reg[thread_y * BLOCK_N_DIM + thread_x] += sum;
}

template<const int WARP_SIZE, const int QUANT_BLOCK_SIZE>
__global__ void gemm_quant_q_8_0_nt(
    int M, int N, int K,
    int a_ld, int b_ld, int c_ld,
    const tff::core::quant::Q_8_0 *__restrict__ a,
    const tff::core::quant::Q_8_0 *__restrict__ b,
    float *__restrict__ c) {
    const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    const int thread_y = g_thread_id / BLOCK_N_DIM;
    const int thread_x = g_thread_id % BLOCK_N_DIM;

    const int start_m = blockIdx.y * BLOCK_M_DIM;
    const int start_n = blockIdx.x * BLOCK_N_DIM;
    const int dst_m_index = start_m + thread_y;
    const int dst_n_index = start_n + thread_x;

    __shared__ tff::core::quant::Q_8_0 a_sm[BLOCK_M_DIM][BLOCK_K_DIM + PAD_SIZE];
    __shared__ tff::core::quant::Q_8_0 b_sm[BLOCK_N_DIM][BLOCK_K_DIM + PAD_SIZE];
    float c_reg[BLOCK_M_DIM][BLOCK_N_DIM] = {0};

    for (size_t k = 0; k < K; k += BLOCK_K_DIM) {
        load_tile_a<QUANT_BLOCK_SIZE>(a_ld, M, thread_x, thread_y, start_m, k, a, &a_sm[0][0]);
        load_tile_b<QUANT_BLOCK_SIZE>(b_ld, N, thread_x, thread_y, start_n, k, b, &b_sm[0][0]);
        __syncthreads();

        compute_tile<QUANT_BLOCK_SIZE>(thread_x, thread_y, &a_sm[0][0], &b_sm[0][0], &c_reg[0][0]);
        __syncthreads();
    }

    if (dst_m_index < M && dst_n_index < N) {
        c[dst_m_index * c_ld + dst_n_index] += c_reg[thread_y][thread_x];
    }
}

static int ggml_cuda_dp4a(const int a, const int b, int c) {
    const int8_t *a8 = (const int8_t *) &a;
    const int8_t *b8 = (const int8_t *) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}

static void quant_q_8_0_gemm_cpu(int M, int N, int K,
                                 int a_ld, int b_ld, int c_ld,
                                 const tff::core::quant::Q_8_0 *a,
                                 const tff::core::quant::Q_8_0 *b,
                                 float *c) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                const auto &a_block = a[m * a_ld + k];
                const auto &b_block = b[n * b_ld + k];

                const float d_a = __half2float(a_block.d);
                const float d_b = __half2float(b_block.d);
                if (m == 0 && n == 0 && k == 0) {
                    printf("d_a: %lf, d_b: %lf\n", d_a, d_b);
                }
                const int *a_k_data = reinterpret_cast<const int32_t *>(&a_block.qs[0]);

                const int *b_k_data = reinterpret_cast<const int32_t *>(&b_block.qs[0]);

                float dot_q = 0;
                for (int i = 0; i < tff::core::quant::Q_8_0::BLOCK_SIZE / 4; ++i) {
                    auto a_value = a_k_data[i];
                    auto b_value = b_k_data[i];
                    dot_q = ggml_cuda_dp4a(a_value, b_value, dot_q);
                }
                sum += d_a * d_b * static_cast<float>(dot_q);
            }
            c[m * c_ld + n] = sum;
        }
    }
}

constexpr int BLOCK_SIZE = 32;
constexpr int WARP_NUM_PER_BLOCK = 8;

template<const int WARP_SIZE, const int BLOCK_SIZE, const int WARP_NUM_PER_BLOCK>
__global__ void quant_q_8_0(const float *__restrict__ src,
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
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
    }

    float d = max_value / 127.0f;
    if (g_dst_index < M * N / BLOCK_SIZE) {
        if (lane_id == 0) {
            dst_ptr[g_dst_index].d = __float2half(d);
        }
        dst_ptr[g_dst_index].qs[lane_id] =
                max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
    }
}

template<const int WARP_SIZE, const int BLOCK_SIZE>
__global__ void quant_q_8_0_2d(const float *__restrict__ src,
                               void *dst, const int M, const int N, const int ld, const int dst_stride_cnt) {
    const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = g_thread_id / WARP_SIZE;
    const int lane_id = g_thread_id % WARP_SIZE;

    const int row = blockIdx.y * blockDim.y + warp_id;
    const int col = blockIdx.x * blockDim.x + lane_id;
    const int start_dst_row = row;
    const int start_dst_col = col / BLOCK_SIZE;
    auto *dst_ptr = static_cast<tff::core::quant::Q_8_0 *>(dst);

    float x = 0.0f;
    float max_value = 0.0f;
    if (col < N && row < M) {
        x = src[row * ld + col];
        max_value = fabsf(x);
    }

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
    }
    float d = max_value / 127.0f;
    const int index = start_dst_row * dst_stride_cnt + start_dst_col;
    if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
        if (lane_id == 0) {
            dst_ptr[index].d = __float2half(d);
        }
        dst_ptr[index].qs[lane_id] = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
    }
}

#include <cstdint>

inline uint32_t div_u32_cpu(uint32_t n, uint32_t magic, int shift) {
    // 计算 (n * magic) 的高32位：等价于 (uint64_t)n * magic >> 32
    uint64_t product = (uint64_t) n * magic;
    uint32_t high_part = (uint32_t) (product >> 32);
    return (high_part + n) >> shift;
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

void quant_q_8_0_1d(const int64_t m, const int64_t n, std::vector<float> &src,
                    std::vector<tff::core::quant::Q_8_0> &dst) {
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
                float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (dst[i * n / BLOCK_SIZE + j].
                                     qs[k]);
                //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
                if (delta_qs > 1.0f) {
                    printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i, j, k, delta_qs);
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
                float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (dst[i * n / BLOCK_SIZE + j].
                                     qs[k]);
                //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
                if (delta_qs > 1.0f) {
                    printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i, j, k, delta_qs);
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

int main(int argc, char *argv[]) {
    cudaDeviceProp device_prop{};
    cudaGetDeviceProperties(&device_prop, 0);
    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-127, 127);
    int m = 2048;
    int n = 2048;
    int k = 2048;
    int quant_k = k / tff::core::quant::Q_8_0::BLOCK_SIZE;

    std::vector<float> a_mat;
    a_mat.resize(m * k);
    PopulateVector<float>(a_mat, mt, dist);
    std::vector<tff::core::quant::Q_8_0> a_quant;
    quant_q_8_0_2d(m, k, a_mat, a_quant);
    tff::core::quant::Q_8_0 *a_mat_gpu = nullptr;
    cudaMalloc(&a_mat_gpu, sizeof(tff::core::quant::Q_8_0) * m * quant_k);
    cudaMemcpy(a_mat_gpu, a_quant.data(), sizeof(tff::core::quant::Q_8_0) * m * quant_k,
               cudaMemcpyHostToDevice);

    std::vector<float> b_mat;
    b_mat.resize(n * k);
    PopulateVector<float>(b_mat, mt, dist);
    std::vector<tff::core::quant::Q_8_0> b_quant;
    quant_q_8_0_2d(n, k, b_mat, b_quant);
    tff::core::quant::Q_8_0 *b_mat_gpu = nullptr;
    cudaMalloc(&b_mat_gpu, sizeof(tff::core::quant::Q_8_0) * n * quant_k);
    cudaMemcpy(b_mat_gpu, b_quant.data(), sizeof(tff::core::quant::Q_8_0) * n * quant_k,
               cudaMemcpyHostToDevice);

    //c
    float *c_mat_gpu = nullptr;
    cudaMalloc(&c_mat_gpu, sizeof(float) * m * n);
    cudaMemset(c_mat_gpu, 0, sizeof(float) * m * n);
    std::vector<float> c_mat;
    c_mat.resize(m * n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 grid((n + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (m + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
    dim3 block(BLOCK_N_DIM, BLOCK_M_DIM, 1);
    printf("grid:%d, %d\n",grid.x, grid.y);
    printf("block:%d, %d\n",block.x, block.y);
    gemm_quant_q_8_0_nt<32, 32><<<grid, block>>>(m, n, quant_k, quant_k, quant_k, n, a_mat_gpu, b_mat_gpu, c_mat_gpu);

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaMemcpy(c_mat.data(), c_mat_gpu, c_mat.size() * sizeof(float),
               cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cudaFree(c_mat_gpu);
    c_mat_gpu = nullptr;
    cudaFree(a_mat_gpu);
    a_mat_gpu = nullptr;
    cudaFree(b_mat_gpu);
    b_mat_gpu = nullptr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Matrix size: %d x %d\n", m, n);
    printf("Kernel time: %.4f ms\n", milliseconds);
#ifdef _DEBUG
    std::vector<float> c_mat_cpu;
    c_mat_cpu.resize(m * n);
    quant_q_8_0_gemm_cpu(m, n, quant_k, quant_k,
                         quant_k, n, a_quant.data(), b_quant.data(), c_mat_cpu.data());
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = c_mat_cpu[i * n + j] - c_mat[i * n + j];
            if (delta > 1.0f) {
                printf("m: %d, n: %d, error: %f, c_mat_cpu[%d]: %f, c_mat[%d]: %f,\n", i, j, delta,
                    i * n + j,c_mat_cpu[i * n + j],i * n + j, c_mat[i * n + j]);
                return 0;
            }
        }
    }
    printf("quant gemm sucess!\n");
#endif
    return 0;
}
