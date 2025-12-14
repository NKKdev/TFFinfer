//
// Created by nkk on 2025/12/13.
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
#include <cooperative_groups.h>
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
constexpr int VEC_DIM_N = 2;
constexpr int VEC_DIM_K = 2;
constexpr int VEC_DIM_M = 8;
constexpr int BLOCK_DIM_M = 64; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_M;
constexpr int BLOCK_DIM_N = 64; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_N;
constexpr int PAD_SIZE = 0; //BLOCK_DIM_K;
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
    } else if (count == 2 && reinterpret_cast<uintptr_t>(addr) % 4 == 0) {
        float2 v = *reinterpret_cast<const float2 *>(addr);
        out[0] = v.x;
        out[1] = v.y;
    }
    else {
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
    }
}

template<>
__device__ __forceinline__ void load_vec<half>(const half *addr, half *out, int count) {
    if (count == 8 && reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
        uint4 v = *reinterpret_cast<const uint4 *>(addr);
        const half *h = reinterpret_cast<const half *>(&v);
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = h[i];
    } else if (count == 2 && reinterpret_cast<uintptr_t>(addr) % 4 == 0) {
        const half2 *h = reinterpret_cast<const half2 *>(addr);
        out[0] = h[0].x;
        out[1] = h[0].y;
    } else {
#pragma unroll
        for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
    }
}

template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
    const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int PAD_SIZE>
__device__ void load_tile_vec_t(const int ld, const int dim,
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
template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
    const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int PAD_SIZE>
__device__ void load_tile_vec_n(const int ld, const int dim,
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
                sm[sm_col * (BLOCK_DIM_K + PAD_SIZE) + (sm_row_base + i)] = val[i];
            }
        }
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
    const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int PAD_SIZE>
__device__ void compute_tile_attention_gemm(const int k_size, const int thread_x, const int warp_id,
                                            T *a_sm, T *b_sm,
                                            float *c_reg) {
#pragma unroll
    for (int kk = 0; kk < k_size; kk++) {

#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            float a_reg = 0.0f;
            a_reg = __half2float(
                    a_sm[(kk) * (BLOCK_DIM_M + PAD_SIZE) + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M]);
#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                float b_reg = 0.0f;
                b_reg = __half2float(
                        b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                c_reg[mm * VEC_DIM_N + nn] += a_reg * b_reg;
            }
        }
    }
}
template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
__device__ void compute_tile_attention_gemm_with_rope(const int k_size, const int thread_x, const int warp_id,
    const int start_m, const int start_n,
                                            T *a_sm, T *b_sm, float *cos_sin_sm,
                                            float *c_reg) {
    auto *q_cos_sin_table_ptr = reinterpret_cast<float2 *>(cos_sin_sm + start_m * BLOCK_DIM_K);
    auto *k_cos_sin_table_ptr = reinterpret_cast<float2 *>(cos_sin_sm + start_n * BLOCK_DIM_K);
#pragma unroll
    for (int kk = 0; kk < k_size; kk += 2) {

#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            float2 a_cos_sin_coef = q_cos_sin_table_ptr[kk / 2 + (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_K + PAD_SIZE) / 2];

            float a_reg[2] = {0.0f};
            a_reg[0] = __half2float(
                    a_sm[(kk) * (BLOCK_DIM_M + PAD_SIZE) + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M]);
            a_reg[1] = __half2float(
                    a_sm[(kk + 1) * (BLOCK_DIM_M + PAD_SIZE) + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M]);

            float tmp = a_reg[0] * a_cos_sin_coef.x - a_reg[1] * a_cos_sin_coef.y;
            a_reg[1] = a_reg[0] * a_cos_sin_coef.y + a_reg[1] * a_cos_sin_coef.x;
            a_reg[0] = tmp;

#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                float2 b_cos_sin_coef = k_cos_sin_table_ptr[kk / 2 + (thread_x + nn * BLOCK_DIM_N / VEC_DIM_N) * (BLOCK_DIM_K + PAD_SIZE) / 2];

                float b_reg[2] = {0.0f};
                b_reg[0] = __half2float(
                        b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                b_reg[1] = __half2float(
                        b_sm[(kk + 1) * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);

                float tmp = b_reg[0] * b_cos_sin_coef.x - b_reg[1] * b_cos_sin_coef.y;
                b_reg[1] = b_reg[0] * b_cos_sin_coef.y + b_reg[1] * b_cos_sin_coef.x;
                b_reg[0] = tmp;
                c_reg[mm * VEC_DIM_N + nn] += a_reg[0] * b_reg[0] + a_reg[1] * b_reg[1];
            }
        }
    }
}
template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int
    BLOCK_DIM_K, const int PAD_SIZE>
__device__ void compute_softmax_pv(const int N, const int v_ld, const int start_m, const int start_n, const int start_d,
                                   const int thread_x, const int warp_id,
                                   const float scale, float *g_max_value, float *g_sum_value, T *sm, float *c_reg,
                                   float *output) {
    const int base_n = start_n + thread_x;
    const int valid_n = min(VEC_DIM_N, (N - base_n + 31) / 32);

    for (int mm = 0; mm < VEC_DIM_M; mm++) {
        const float old_max = g_max_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
        const float old_sum = g_sum_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
        float max_value = {-1e20f};
        float sum_value = {0.0f};
        float new_max_value = {-1e20f};
        float new_sum_value = {0.0f};

        for (int nn = 0; nn < valid_n; nn++) {
            c_reg[mm * VEC_DIM_N + nn] *= scale;
            max_value = fmaxf(max_value, c_reg[mm * VEC_DIM_N + nn]);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, 32));
        }
        //
        for (int nn = 0; nn < valid_n; nn++) {
            c_reg[mm * VEC_DIM_N + nn] = expf(c_reg[mm * VEC_DIM_N + nn] - max_value);
            sum_value += c_reg[mm * VEC_DIM_N + nn];
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum_value += __shfl_xor_sync(0xffffffff, sum_value, offset, 32);
        }

        new_max_value = fmaxf(max_value, old_max);
        new_sum_value = sum_value * expf(max_value - new_max_value) + old_sum * expf(
                            old_max - new_max_value);

        for (int kk = start_d; kk < BLOCK_DIM_K; kk++) {
            float acc = 0.0f;
            for (int nn = 0; nn < valid_n; nn++) {
                float v_reg = __half2float(
                    sm[kk * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
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

            output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] =
            (output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] * old_sum * expf(
                 old_max - new_max_value) +
             acc * expf(max_value - new_max_value)) / new_sum_value;
        }
        g_max_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_max_value;
        g_sum_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_sum_value;
    }
}

template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
    const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE,
    const int rope_flag>
__global__ void flash_attention(const int M, const int N, const int D,
                                const int q_ld, const int k_ld, const int v_ld,
                                const float scale,
                                const int num_q_heads, const int num_kv_heads,
                                const T *__restrict__ q_global,
                                const T *__restrict__ k_global,
                                const T *__restrict__ v_global,
                                float *__restrict__ cos_sin_table,
                                float *__restrict__ max_value_global,
                                float *__restrict__ sum_value_global,
                                float *out_put) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int ld_thread_block_n = BLOCK_DIM_K / VEC_DIM_K;
    const int thread_x = thread_id % ld_thread_block_n;
    const int warp_id = thread_id / ld_thread_block_n;

    const int block_x = blockIdx.x;
    const int start_m = block_x * BLOCK_DIM_M;

    const T *q = q_global + (blockIdx.z * num_q_heads + blockIdx.y) * M * D;
    const int kv_group_per_q = num_q_heads / num_kv_heads;
    const int kv_group_start_index = blockIdx.y / kv_group_per_q;
    const T *k = k_global + (blockIdx.z * num_kv_heads + kv_group_start_index) * N * D;
    const T *v = v_global + (blockIdx.z * num_kv_heads + kv_group_start_index) * N * D;
    float *output = out_put + (blockIdx.z * num_q_heads + blockIdx.y) * M * D;
    float *g_max_value = max_value_global + (blockIdx.z * num_q_heads + blockIdx.y) * M;
    float *g_sum_value = sum_value_global + (blockIdx.z * num_q_heads + blockIdx.y) * M;


    constexpr int shared_mem_block_size = BLOCK_DIM_K * (BLOCK_DIM_M + PAD_SIZE);
    __shared__ T sm[shared_mem_block_size * 3];//2 * kv + 1 * q;(k, v共享一块shared mem)

    int flip_flag = 0;
    load_tile_vec_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
        q_ld, M, thread_x, warp_id, start_m, 0,
        q, &sm[shared_mem_block_size * 2]);
    load_tile_vec_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
        k_ld, N, thread_x, warp_id, 0, 0,
        k, &sm[(flip_flag) * shared_mem_block_size]);
    __syncthreads();

    const int n_stage = (N + BLOCK_DIM_N - 1) / BLOCK_DIM_N;

    for (int n = 0; n < n_stage; n++) {
        int n_start = n * BLOCK_DIM_N;
        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        int d = 0;
        const int k_size = min(BLOCK_DIM_K, D - d);
        if (rope_flag == 1) {
            compute_tile_attention_gemm_with_rope<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                        k_size, thread_x, warp_id, start_m, n_start, &sm[shared_mem_block_size * 2], &sm[(flip_flag) * shared_mem_block_size],
                        cos_sin_table, c_reg);
        }else {
            compute_tile_attention_gemm<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
            k_size, thread_x, warp_id, &sm[shared_mem_block_size * 2], &sm[(flip_flag) * shared_mem_block_size], c_reg);
        }


        const int next_n = (n + 1) * BLOCK_DIM_N;
        if (next_n < N) {
            load_tile_vec_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
                k_ld, N, thread_x, warp_id, next_n, 0,
                k, &sm[(1 - flip_flag) * shared_mem_block_size]);
        }
        __syncthreads();

        load_tile_vec_t<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            v_ld, N, thread_x, warp_id, n_start, d,
            v, &sm[(flip_flag) * shared_mem_block_size]);
        __syncthreads();

        compute_softmax_pv<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
            N, v_ld, start_m, n_start, d, thread_x, warp_id, scale, g_max_value,
            g_sum_value, &sm[(flip_flag) * shared_mem_block_size], &c_reg[0], output);
        __syncthreads();
        flip_flag ^= 1;
    }
}

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
template<const int rope_flag>
static void apply_rope_to_vec(
    const T *src,
    float *dst,
    int head_dim,
    int pos,
    const float *cos_ptr,
    const float *sin_ptr) {
    int half = head_dim / 2;
    for (int i = 0; i < half; ++i) {
        float x0 = to_float(src[2 * i]);
        float x1 = to_float(src[2 * i + 1]);
        if (rope_flag == 1) {
            float c = cos_ptr[pos * half + i];
            float s = sin_ptr[pos * half + i];
            dst[2 * i] = x0 * c - x1 * s;
            dst[2 * i + 1] = x0 * s + x1 * c;
            int tmp = i + 1;
        }else {
            dst[2 * i] = x0 ;
            dst[2 * i + 1] = x1;
        }

    }
    //printf("\n");
}

template<typename T, const int rope_flag>
void flash_attention_cpu_single_head(
    int m, int n, int k,
    const T *q_mat,
    const T *k_mat,
    const T *v_mat,
    float *out_ptr,
    const float *cos_ptr,
    const float *sin_ptr,
    int block_size_m = 64,
    int block_size_n = 64) {
    const float scaling = 1.0f / std::sqrt(static_cast<float>(k));
    std::vector<float> q_rot(k);
    std::vector<float> k_rot(k);

    // for (int im = 0; im < 64; ++im) {
    //     for (int kk = 0; kk < 32; ++kk) {
    //         if (im == 0) {
    //             printf("cpu mm: %d, kk: %d, coff cos: %lf, coff sin: %lf \n", im, kk, cos_ptr[im * 32 + kk], sin_ptr[im * 32 + kk]);
    //         }
    //
    //     }
    //     if (im == 0) {
    //         printf("\n");
    //     }
    // }
    for (int m_start = 0; m_start < m; m_start += block_size_m) {
        int m_end = std::min(m_start + block_size_m, m);
        int current_m = m_end - m_start;

        std::vector<float> running_max(current_m, -1e20f);
        std::vector<float> running_sum(current_m, 0.0f);

        for (int n_start = 0; n_start < n; n_start += block_size_n) {
            int n_end = std::min(n_start + block_size_n, n);
            int current_n = n_end - n_start;

            std::vector<std::vector<float> > local_logits(current_m, std::vector<float>(current_n, 0.0f));

            for (int im = 0; im < current_m; ++im) {
                int i = m_start + im;
                apply_rope_to_vec<rope_flag>(q_mat + i * k, q_rot.data(), k, i, cos_ptr, sin_ptr);

                for (int j_idx = 0; j_idx < current_n; ++j_idx) {
                    int j = n_start + j_idx;
                    apply_rope_to_vec<rope_flag>(k_mat + j * k, k_rot.data(), k, j, cos_ptr, sin_ptr);

                    float dot = 0.0f;
                    for (int l = 0; l < k; ++l) {
                        dot += q_rot[l] * k_rot[l];
                    }
                    local_logits[im][j_idx] = dot * scaling;
                }
            }

            std::vector<float> local_max(current_m);
            std::vector<float> local_sum(current_m);
            std::vector<std::vector<float> > local_exps(current_m, std::vector<float>(current_n));

            for (int im = 0; im < current_m; ++im) {
                local_max[im] = *std::max_element(local_logits[im].begin(), local_logits[im].end());
                local_sum[im] = 0.0f;
                for (int j_idx = 0; j_idx < current_n; ++j_idx) {
                    local_exps[im][j_idx] = ::expf(local_logits[im][j_idx] - local_max[im]);
                    local_sum[im] += local_exps[im][j_idx];
                }
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
                    float corrected_acc = (exp_old * out_ptr[i * k + l] + acc * exp_local);
                    out_ptr[i * k + l] = corrected_acc;
                }

                running_max[im] = new_max;
                running_sum[im] = new_sum;
            }
        }
        for (int mm = 0; mm < current_m; ++mm) {
            int i = m_start + mm;
            if (running_sum[mm] != 0.0f) {
                for (int l = 0; l < k; ++l) {
                    out_ptr[i * k + l] /= running_sum[mm];
                }
            }
        }
    }
}

template<typename T, const int rope_flag>
std::vector<float> flash_attention_cpu_gqa(
    int batch,
    int m, int n, int head_dim,
    int num_q_heads,
    int num_kv_heads,
    const std::vector<T> &q_mat,
    const std::vector<T> &k_mat,
    const std::vector<T> &v_mat,
    const std::vector<float> &cos_table,
    const std::vector<float> &sin_table,
    int block_size_m = 64,
    int block_size_n = 64) {
    if (num_q_heads % num_kv_heads != 0) return {};
    int q_per_kv = num_q_heads / num_kv_heads;

    size_t q_size = static_cast<size_t>(batch) * num_q_heads * m * head_dim;
    size_t kv_size = static_cast<size_t>(batch) * num_kv_heads * n * head_dim;
    if (q_mat.size() != q_size || k_mat.size() != kv_size || v_mat.size() != kv_size) {
        return {};
    }

    int half_dim = head_dim / 2;
    if (cos_table.size() < static_cast<size_t>(std::max(m, n)) * half_dim ||
        sin_table.size() < static_cast<size_t>(std::max(m, n)) * half_dim) {
        return {};
    }

    std::vector<float> output(batch * num_q_heads * m * head_dim);

    for (int b = 0; b < batch; ++b) {
        for (int qh = 0; qh < num_q_heads; ++qh) {
            int kvh = qh / q_per_kv;

            const T *q_ptr = q_mat.data() + ((b * num_q_heads + qh) * m * head_dim);
            const T *k_ptr = k_mat.data() + ((b * num_kv_heads + kvh) * n * head_dim);
            const T *v_ptr = v_mat.data() + ((b * num_kv_heads + kvh) * n * head_dim);
            float *out_ptr = output.data() + ((b * num_q_heads + qh) * m * head_dim);

            flash_attention_cpu_single_head<T, rope_flag>(
                m, n, head_dim,
                q_ptr, k_ptr, v_ptr,
                out_ptr,
                cos_table.data(),
                sin_table.data(),
                block_size_m,
                block_size_n
            );
        }
    }

    return output;
}

std::vector<float> precompute_rope_tables(int max_seq_len, int dim, float base = 10000.0f) {
    std::vector<float> inv_freq(dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(base, float(2 * i) / dim);
    }

    std::vector<float> table(max_seq_len * (dim / 2));
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < dim / 2; ++i) {
            float freq = pos * inv_freq[i];
            table[pos * (dim / 2) + i] = freq;
        }
    }
    return table;
}

// 然后：

template<typename T, const int rope_flag>
void flash_attention(int batch,
                     int M, int N, int D,
                     int num_q_heads,
                     int num_kv_heads,
                     const std::vector<T> &q_mat,
                     const std::vector<T> &k_mat,
                     const std::vector<T> &v_mat,
                     const std::vector<float> &cos_table,
                     const std::vector<float> &sin_table,
                     int block_size_m = 64,
                     int block_size_n = 64) {
    T *q_gpu = nullptr;
    cudaMalloc((void **) &q_gpu, sizeof(T) * batch * num_q_heads * M * D);
    cudaMemcpy(q_gpu, q_mat.data(), sizeof(T) * batch * num_q_heads * M * D, cudaMemcpyHostToDevice);
    T *k_gpu = nullptr;
    cudaMalloc((void **) &k_gpu, sizeof(T) * batch * num_kv_heads * N * D);
    cudaMemcpy(k_gpu, k_mat.data(), sizeof(T) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
    T *v_gpu = nullptr;
    cudaMalloc((void **) &v_gpu, sizeof(T) * batch * num_kv_heads * N * D);
    cudaMemcpy(v_gpu, v_mat.data(), sizeof(T) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
    float *max_value = nullptr;
    cudaMalloc((void **) &max_value, sizeof(float) * batch * num_q_heads * M);
    cudaMemset(max_value, -MAXFLOAT, sizeof(float) * M);
    float *sum_value = nullptr;
    cudaMalloc((void **) &sum_value, sizeof(float) * batch * num_q_heads * M);
    cudaMemset(sum_value, 0, sizeof(float) * batch * num_q_heads * M);
    std::vector<float> cos_sin_table(M * D);
    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < D; d += 2) {
            cos_sin_table[m * D + d] = cos_table[m * D / 2 + d / 2];
            cos_sin_table[m * D + d + 1] = sin_table[m * D / 2 + d / 2];
            // if (m == 1)
            //     {
            //     printf("cpu mm: %d, kk: %d, coff cos: %lf, coff sin: %lf \n", m, d, cos_sin_table[m * D + d], cos_sin_table[m * D + d + 1]);
            // }
        }
        // if (m == 1)
        //     {
        //     printf("\n");
        // }
    }

    float *cos_sin_table_gpu = nullptr;
    cudaMalloc((void **) &cos_sin_table_gpu, sizeof(float) * M * D);
    cudaMemcpy(cos_sin_table_gpu, cos_sin_table.data(), sizeof(float) * M * D, cudaMemcpyHostToDevice);

    float *output = nullptr;
    cudaMalloc((void **) &output, sizeof(float) * batch * num_q_heads * M * D);
    cudaMemset(output, 0, sizeof(float) * batch * num_q_heads * M * D);
    const int block_num = max((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, 1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, num_q_heads, batch);
    dim3 block(256);
    printf("grid x: %d, grid y: %d, grid z: %d \n", grid.x, grid.y, grid.z);
    printf("block x: %d, block y: %d\n", block.x, block.y);

    flash_attention<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, rope_flag><<<grid
            , block>>>(
                M, N, D,
                D, D, D,
                (1.0f / std::sqrt(static_cast<float>(D))),
                num_q_heads,
                num_kv_heads,
                q_gpu, k_gpu, v_gpu, cos_sin_table_gpu,
                max_value, sum_value, output);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::vector<float> output_cpu;
    output_cpu.resize(batch * num_q_heads * M * D);
    cudaMemcpy(output_cpu.data(), output, sizeof(float) * batch * num_q_heads * M * D, cudaMemcpyDeviceToHost);

    cudaFree(q_gpu);
    cudaFree(k_gpu);
    cudaFree(v_gpu);
    cudaFree(max_value);
    cudaFree(sum_value);
    cudaFree(output);
    cudaFree(cos_sin_table_gpu);

    if (rope_flag == 1) {
        const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch *
            num_q_heads * M * N + batch * num_q_heads * M * (D / 2) * (4 + 2);
        const double gflops = static_cast<double>(flops) / 1e9;
        const double seconds = milliseconds / 1000.0;
        const double gflops_per_sec = gflops / seconds;
        printf("******************************************\n");
        printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
        printf("Kernel time: %.4f ms\n", milliseconds);
        printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
        printf("手写 flash_attention rope Performance: %.2f GFLOPS/s\n", gflops_per_sec);
    }else {
        const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch *
                            num_q_heads * M * N;
        const double gflops = static_cast<double>(flops) / 1e9;
        const double seconds = milliseconds / 1000.0;
        const double gflops_per_sec = gflops / seconds;
        printf("******************************************\n");
        printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
        printf("Kernel time: %.4f ms\n", milliseconds);
        printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
        printf("手写 flash_attention Performance: %.2f GFLOPS/s\n", gflops_per_sec);
    }

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

    std::vector<float> cpu_result = flash_attention_cpu_gqa<T, rope_flag>(batch, M, N, D, num_q_heads, num_kv_heads, q_mat, k_mat,
                                                            v_mat, cos_table, sin_table);
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
    for (int b = 0; b < batch; b++) {
        for (int q = 0; q < num_q_heads; q++) {
            float *cpu_result_single = cpu_result.data() + (b * num_q_heads + q) * M * D;
            float *gpu_result_single = output_cpu.data() + (b * num_q_heads + q) * M * D;
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < D; ++i) {
                    float delta = cpu_result_single[j * D + i] - gpu_result_single[j * D + i];
                    if (fabs(delta) > 0.01f) {
                        printf("error : %f, m: %d, d: %d\n", delta, j, i);
                        return;
                    }
                }
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
    int dim = 1024;
    int m = dim;
    int n = dim;
    int k = 64;
    int num_q_heads = 15;
    int num_kv_heads = 5;
    int batch = 2;


    std::vector<T> a_mat;
    a_mat.resize(batch * num_q_heads * m * k);
    std::vector<T> b_mat;
    b_mat.resize(batch * num_kv_heads * n * k);
    std::vector<T> c_mat;
    c_mat.resize(batch * num_kv_heads * n * k);

    PopulateVector<T>(a_mat, mt, dist);
    PopulateVector<T>(b_mat, mt, dist);
    PopulateVector<T>(c_mat, mt, dist);
    auto angles = precompute_rope_tables(std::max(m, n), k);
    std::vector<float> cos_table(angles.size()), sin_table(angles.size());
    for (size_t i = 0; i < angles.size(); ++i) {
        cos_table[i] = std::cos(angles[i]);
        sin_table[i] = std::sin(angles[i]);
    }

    flash_attention<T, 1>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat, cos_table, sin_table);

#else
    int dim = 1024;
    int m = dim;
    int n = dim;
    int k = 64;
    int num_q_heads = 15;
    int num_kv_heads = 5;
    int batch = 2;


    std::vector<T> a_mat;
    a_mat.resize(batch * num_q_heads * m * k);
    std::vector<T> b_mat;
    b_mat.resize(batch * num_kv_heads * n * k);
    std::vector<T> c_mat;
    c_mat.resize(batch * num_kv_heads * n * k);

    PopulateVector<T>(a_mat, mt, dist);
    PopulateVector<T>(b_mat, mt, dist);
    PopulateVector<T>(c_mat, mt, dist);
    flash_attention<T>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat);
#endif


    return 0;
}
