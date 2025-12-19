//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {
    template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int PAD_SIZE, const int ELEMENTS_PER_LOAD>
    __device__ __forceinline__ void load_tile_vec_n(const int ld, const int dim,
                                                    const int thread_x, const int warp_id,
                                                    const int start_m,
                                                    const int k,
                                                    const T *__restrict__ global_mem,
                                                    T *sm) {
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                const int dim1 = k + thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                                 ELEMENTS_PER_LOAD;
                T val[ELEMENTS_PER_LOAD] = {0};
                if (dim0_base < dim) {
                    const int actual_load = min(ELEMENTS_PER_LOAD, ld - dim1);
                    if (actual_load > 0) {
                        load_vec<T>(&global_mem[dim0_base * ld + dim1], val, actual_load);
                    }
                }
                const int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
                const int sm_col_base = thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                                        ELEMENTS_PER_LOAD;
#pragma unroll
                for (int i = 0; i < ELEMENTS_PER_LOAD; ++i) {
                    sm[sm_row * (BLOCK_DIM_K + PAD_SIZE) + (sm_col_base + i)] = val[i];
                }
            }
        }
    }


    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __device__ __forceinline__ void compute_tile_attention_gemm(const int N, const int k_size, const int thread_x, const int warp_id,
                                                                const int start_m, const int start_n, T *a_sm, T *b_sm,const T *mask,
                                                                float *c_reg) {
        auto *q_sm_ptr = reinterpret_cast<half2 *>(a_sm);
        auto *k_sm_ptr = reinterpret_cast<half2 *>(b_sm);
#pragma unroll
        for (int kk = 0; kk < k_size; kk += 2) {
            half2 a_reg;
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                a_reg = q_sm_ptr[(kk / 2) + (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_K + PAD_SIZE) / 2];
                half2 b_reg;
#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N; nn++) {
                    b_reg = k_sm_ptr[(kk / 2) + (thread_x + nn * BLOCK_DIM_N / VEC_DIM_N) * (BLOCK_DIM_K + PAD_SIZE) /
                                     2];

                    c_reg[mm * VEC_DIM_N + nn] += __half2float(
                        __hadd(__hmul(a_reg.x, b_reg.x), __hmul(a_reg.y, b_reg.y)));
                }
            }
        }

#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                const float mask_value = __ldg(&mask[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * N + start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                c_reg[mm * VEC_DIM_N + nn] += mask_value;
            }
        }
    }


    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __device__ __forceinline__ void compute_tile_attention_gemm_with_rope(
        const int N, const int k_size, const int thread_x,
        const int warp_id,
        const int start_m, const int start_n,
        T *a_sm, T *b_sm, float *cos_sin_sm, const T *mask,
        float *c_reg) {

        auto *q_cos_sin_table_ptr = reinterpret_cast<float2 *>(cos_sin_sm + start_m * BLOCK_DIM_K);
        auto *k_cos_sin_table_ptr = reinterpret_cast<float2 *>(cos_sin_sm + start_n * BLOCK_DIM_K);
        auto *q_sm_ptr = reinterpret_cast<half2 *>(a_sm);
        auto *k_sm_ptr = reinterpret_cast<half2 *>(b_sm);
#pragma unroll
        for (int kk = 0; kk < k_size; kk += 2) {
            float2 a_cos_sin_coef;
            half2 a_reg;
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                a_cos_sin_coef = q_cos_sin_table_ptr[
                    kk / 2 + (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_K) / 2];

                a_reg = q_sm_ptr[(kk / 2) + (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_K + PAD_SIZE) / 2];

                half2 q_rot = complex_mul_half2(
                    a_reg,
                    make_half2(__float2half(a_cos_sin_coef.x), __float2half(a_cos_sin_coef.y)));
                float2 b_cos_sin_coef;
                half2 b_reg;
#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N; nn++) {
                    b_cos_sin_coef = k_cos_sin_table_ptr[
                        kk / 2 + (thread_x + nn * BLOCK_DIM_N / VEC_DIM_N) * (BLOCK_DIM_K) / 2];

                    b_reg = k_sm_ptr[(kk / 2) + (thread_x + nn * BLOCK_DIM_N / VEC_DIM_N) * (BLOCK_DIM_N + PAD_SIZE) /
                                     2];
                    half2 k_rot = complex_mul_half2(
                        b_reg,
                        make_half2(__float2half(b_cos_sin_coef.x), __float2half(b_cos_sin_coef.y)));

                    c_reg[mm * VEC_DIM_N + nn] += __half2float(
                        __hadd(__hmul(q_rot.x, k_rot.x), __hmul(q_rot.y, k_rot.y)));
                }
            }
        }

#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                const float mask_value = __ldg(&mask[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * N + start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                c_reg[mm * VEC_DIM_N + nn] += mask_value;
            }
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const
        int
        BLOCK_DIM_K, const int PAD_SIZE>
    __device__ __forceinline__ void compute_softmax_pv(const int N, const int v_ld, const int start_m,
                                                       const int start_n, const int start_d,
                                                       const int thread_x, const int warp_id,
                                                       const float scale, float *g_max_value, float *g_sum_value, T *sm,
                                                       float *c_reg,
                                                       float *output) {
        const int base_n = start_n + thread_x;
        const int valid_n = min(VEC_DIM_N, (N - base_n + 31) / 32);
        auto *v_sm_ptr = reinterpret_cast<half2 *>(sm);
#pragma unroll
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

            for (int kk = start_d; kk < BLOCK_DIM_K; kk += 2) {
                float acc[2] = {0.0f};
                for (int nn = 0; nn < valid_n; nn++) {
                    float2 v_reg = __half22float2(
                        v_sm_ptr[kk / 2 + (thread_x + nn * BLOCK_DIM_N / VEC_DIM_N) * (BLOCK_DIM_K + PAD_SIZE) / 2]);
                    acc[0] += v_reg.x * c_reg[mm * VEC_DIM_N + nn];
                    acc[1] += v_reg.y * c_reg[mm * VEC_DIM_N + nn];
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    acc[0] += __shfl_xor_sync(0xffffffff, acc[0], offset, 32);
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    acc[1] += __shfl_xor_sync(0xffffffff, acc[1], offset, 32);
                }

                output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] =
                (output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] * old_sum * expf(
                     old_max - new_max_value) +
                 acc[0] * expf(max_value - new_max_value)) / new_sum_value;
                output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk + 1] =
                (output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk + 1] * old_sum * expf(
                     old_max - new_max_value) +
                 acc[1] * expf(max_value - new_max_value)) / new_sum_value;
            }
            g_max_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_max_value;
            g_sum_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_sum_value;
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE,
        const int ELEMENTS_PER_LOAD>
    __global__ void flash_attention_fp16_64(const int M, const int N, const int D,
                                            const int q_ld, const int k_ld, const int v_ld,
                                            const float scale,
                                            const int num_q_heads, const int num_kv_heads,
                                            const T *__restrict__ q_global,
                                            const T *__restrict__ k_global,
                                            const T *__restrict__ v_global,
                                            const T *__restrict__ mask,
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


        constexpr int shared_mem_block_size = (BLOCK_DIM_K + PAD_SIZE) * (BLOCK_DIM_M);
        __shared__ T sm[shared_mem_block_size * 3]; //2 * kv + 1 * q;(k, v共享一块shared mem)

        int flip_flag = 0;
        load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
            q_ld, M, thread_x, warp_id, start_m, 0,
            q, &sm[shared_mem_block_size * 2]);
        load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
            k_ld, N, thread_x, warp_id, 0, 0,
            k, &sm[(flip_flag) * shared_mem_block_size]);
        __syncthreads();

        const int n_stage = (N + BLOCK_DIM_N - 1) / BLOCK_DIM_N;

        for (int n = 0; n < n_stage; n++) {
            int n_start = n * BLOCK_DIM_N;
            float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

            int d = 0;
            const int k_size = min(BLOCK_DIM_K, D - d);
            compute_tile_attention_gemm<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
                PAD_SIZE>(
                N, k_size, thread_x, warp_id, start_m, n_start, &sm[shared_mem_block_size * 2], &sm[(flip_flag) * shared_mem_block_size],
                mask, c_reg);

            const int next_n = (n + 1) * BLOCK_DIM_N;
            if (next_n < N) {
                load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
                    k_ld, N, thread_x, warp_id, next_n, 0,
                    k, &sm[(1 - flip_flag) * shared_mem_block_size]);
            }
            __syncthreads();

            load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
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

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE,
        const int ELEMENTS_PER_LOAD>
    __global__ void flash_attention_fp16_64_rope(const int M, const int N, const int D,
                                                 const int q_ld, const int k_ld, const int v_ld,
                                                 const float scale,
                                                 const int num_q_heads, const int num_kv_heads,
                                                 const T *__restrict__ q_global,
                                                 const T *__restrict__ k_global,
                                                 const T *__restrict__ v_global,
                                                 const T *__restrict__ mask,
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


        constexpr int shared_mem_block_size = (BLOCK_DIM_K + PAD_SIZE) * (BLOCK_DIM_M);
        __shared__ T sm[shared_mem_block_size * 3]; //2 * kv + 1 * q;(k, v共享一块shared mem)

        int flip_flag = 0;
        load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
            q_ld, M, thread_x, warp_id, start_m, 0,
            q, &sm[shared_mem_block_size * 2]);
        load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
            k_ld, N, thread_x, warp_id, 0, 0,
            k, &sm[(flip_flag) * shared_mem_block_size]);
        __syncthreads();

        const int n_stage = (N + BLOCK_DIM_N - 1) / BLOCK_DIM_N;

        for (int n = 0; n < n_stage; n++) {
            int n_start = n * BLOCK_DIM_N;
            if (n_start > (start_m + BLOCK_DIM_M)) {
                continue;
            }
            float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

            int d = 0;
            const int k_size = min(BLOCK_DIM_K, D - d);
            compute_tile_attention_gemm_with_rope<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
                PAD_SIZE>(
                N, k_size, thread_x, warp_id, start_m, n_start, &sm[shared_mem_block_size * 2],
                &sm[(flip_flag) * shared_mem_block_size], cos_sin_table, mask, c_reg);


            const int next_n = (n + 1) * BLOCK_DIM_N;
            if (next_n < N) {
                load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
                    k_ld, N, thread_x, warp_id, next_n, 0,
                    k, &sm[(1 - flip_flag) * shared_mem_block_size]);
            }
            __syncthreads();

            load_tile_vec_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE, ELEMENTS_PER_LOAD>(
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
    void flash_attention_64_rope(const int batch, const int M, const int N, const int D,
                                 const int q_ld, const int k_ld, const int v_ld,
                                 const float scale,
                                 const int num_q_heads, const int num_kv_heads,
                                 const T *q_gpu,
                                 const T *k_gpu,
                                 const T *v_gpu,
                                 const T *mask,
                                 float *cos_sin_table,
                                 float *out_put) {
        if (std::is_same_v<T, half>) {
            constexpr int ELEMENTS_PER_LOAD = 2;
            constexpr int THREAD_BLOCK_SIZE = 256;
            constexpr int BLOCK_DIM_K = 64;
            constexpr int VEC_DIM_N = 2;
            constexpr int VEC_DIM_K = 2;
            constexpr int VEC_DIM_M = 8;
            constexpr int BLOCK_DIM_M = 64;
            constexpr int BLOCK_DIM_N = 64;
            constexpr int PAD_SIZE = 1;
            dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, num_q_heads, batch);
            dim3 block(THREAD_BLOCK_SIZE);
            float *max_value_global;
            float *sum_value_global;

            flash_attention_fp16_64_rope<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
                PAD_SIZE,
                ELEMENTS_PER_LOAD><<<grid, block>>>(
                M, N, D, D, D, D,
                (1.0f / std::sqrt(static_cast<float>(D))),
                num_q_heads,
                num_kv_heads,
                q_gpu, k_gpu, v_gpu, mask, cos_sin_table, max_value_global, sum_value_global, out_put);
        } else if (std::is_same_v<T, float>) {
            //todo
        }
    }

    template<typename T>
    void flash_attention_64(const int batch, const int M, const int N, const int D,
                            const int q_ld, const int k_ld, const int v_ld,
                            const float scale,
                            const int num_q_heads, const int num_kv_heads,
                            const T *q_gpu,
                            const T *k_gpu,
                            const T *v_gpu,
                            const T *mask,
                            float *out_put) {
        if (std::is_same_v<T, half>) {
            constexpr int ELEMENTS_PER_LOAD = 2;
            constexpr int THREAD_BLOCK_SIZE = 256;
            constexpr int BLOCK_DIM_K = 64;
            constexpr int VEC_DIM_N = 2;
            constexpr int VEC_DIM_K = 2;
            constexpr int VEC_DIM_M = 8;
            constexpr int BLOCK_DIM_M = 64;
            constexpr int BLOCK_DIM_N = 64;
            constexpr int PAD_SIZE = 2;
            dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, num_q_heads, batch);
            dim3 block(THREAD_BLOCK_SIZE);
            float *max_value_global;
            float *sum_value_global;

            flash_attention_fp16_64<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE,
                ELEMENTS_PER_LOAD><<<grid, block>>>(
                M, N, D, D, D, D,
                (1.0f / std::sqrt(static_cast<float>(D))),
                num_q_heads,
                num_kv_heads,
                q_gpu, k_gpu, v_gpu, mask, max_value_global, sum_value_global, out_put);
        } else if (std::is_same_v<T, float>) {
            //todo
        }
    }

    //
    template<typename T>
    void tff::kernel::FlashAttn<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), FlashAttn<T>::get_op_name().c_str());
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            1, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            2, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(3, para_ptr);
        auto rope_flag = get_param_value<const int>(4, para_ptr);
        if (rope_flag == 1) {
            if (input_tensors.size() != 4) {
                tff::log::Logger::error("memcpy kernel param is invalid!");
                return;
            }
        } else {
            if (input_tensors.size() != 3) {
                tff::log::Logger::error("memcpy kernel param is invalid!");
                return;
            }
        }

        if (output_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        auto q_tensor = input_tensors.at(0);
        auto k_tensor = input_tensors.at(1);
        auto v_tensor = input_tensors.at(2);
        auto mask_tensor = input_tensors.at(3);
        auto output = output_tensors.at(0);
        const int num_q_heads = q_tensor->get_shape()[2];
        const int num_kv_heads = k_tensor->get_shape()[2];
        const int M = q_tensor->get_shape()[1];
        const int N = k_tensor->get_shape()[1];
        const int D = q_tensor->get_shape()[0];
        const int B = q_tensor->get_shape()[3];
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        switch (D) {
            case 64:
                if (rope_flag == 1) {
                    auto *cos_sin_table = static_cast<float *>(input_tensors.at(3)->get_buffer()->ptr());
                    flash_attention_64_rope<T>(B, M, N, D, D, D, D, scale, num_q_heads, num_kv_heads,
                                               static_cast<T *>(q_tensor->get_buffer()->ptr()),
                                               static_cast<T *>(k_tensor->get_buffer()->ptr()),
                                               static_cast<T *>(v_tensor->get_buffer()->ptr()),
                                               static_cast<T *>(mask_tensor->get_buffer()->ptr()),
                                               cos_sin_table,
                                               static_cast<float *>(output->get_buffer()->ptr()));
                } else {
                    flash_attention_64<T>(B, M, N, D, D, D, D, scale, num_q_heads, num_kv_heads,
                                          static_cast<T *>(q_tensor->get_buffer()->ptr()),
                                          static_cast<T *>(k_tensor->get_buffer()->ptr()),
                                          static_cast<T *>(v_tensor->get_buffer()->ptr()),
                                          static_cast<T *>(mask_tensor->get_buffer()->ptr()),
                                          static_cast<float *>(output->get_buffer()->ptr()));
                }

                break;
            case 128:
                break;
            case 256:
                break;
            default:
                break;
        }
    }

    template<typename T>
    std::string tff::kernel::FlashAttn<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;

        return name;
    }

    //template class tff::kernel::FlashAttn<float>;//todo
    template class tff::kernel::FlashAttn<half>;
    //REGISTER_OP_OBJECT(FlashAttn, float);

    REGISTER_OP_OBJECT(FlashAttn, half);
}
