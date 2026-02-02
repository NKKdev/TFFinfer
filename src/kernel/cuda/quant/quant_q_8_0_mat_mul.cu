//
// Created by nkk on 2026/1/3.
//

#include "../../../../cmake-build-release/_deps/fmt-src/include/fmt/os.h"
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {
    template<typename T1, typename T2>
    __device__ __forceinline__ void load_vec(const T1 *addr, T2 *out, const int count);

    template<>
    __device__ __forceinline__ void load_vec<half, half2>(const half *addr, half2 *out, const int count) {
        if (count == 4) {
            const auto *h = reinterpret_cast<const half2 *>(addr);
            out[0] = h[0];
            out[1] = h[1];
            out[2] = h[2];
            out[3] = h[3];
        } else {
            const auto *h = reinterpret_cast<const half2 *>(addr);
            out[0] = h[0];
        }
    }

    template<>
    __device__ __forceinline__ void load_vec<float, float2>(const float *addr, float2 *out, const int count) {
        if (count == 4) {
            const float2 *h = reinterpret_cast<const float2 *>(addr);
            out[0] = h[0];
            out[1] = h[1];
            out[2] = h[2];
            out[3] = h[3];
        } else {
            const float2 *h = reinterpret_cast<const float2 *>(addr);
            out[0] = h[0];
        }
    }

    template<>
    __device__ __forceinline__ void load_vec<int8_t, int32_t>(const int8_t *addr, int32_t *out, const int count) {
        if (count == 4 && reinterpret_cast<uintptr_t>(addr) % 4 == 0) {
            *out = reinterpret_cast<const int32_t>(addr);
        } else {
            int32_t packed = 0;
#pragma unroll
            for (int i = 0; i < count; ++i) {
                packed |= static_cast<int32_t>(static_cast<uint8_t>(addr[i])) << (i * 8);
            }
            out[0] = (packed);
        }
    }

    template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
        const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE, const int ALIGNED_FLAG>
    __device__ void load_tile(const int ld, const int dim,
                              const int thread_x, const int warp_id,
                              const int start_block,
                              const int k,
                              const tff::core::quant::Q_8_0 *__restrict__ global_mem,
                              int *quant_sm, half *scale_sm) {
        const int quant_block_id = thread_x / THREAD_NUM_PER_QUANT_BLOCK;
        const int block_inter_index = thread_x % THREAD_NUM_PER_QUANT_BLOCK;

        //load quant data;
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            int dim0 = start_block + warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD;
#pragma unroll
            for (int kk = 0; kk < VEC_DIM_K; ++kk) {
                int dim1 = k + quant_block_id + kk * QUANT_BLKS_PER_WARP / VEC_DIM_K;
                if (dim1 < ld && dim0 < dim) {
                    int32_t quant_data = 0;
                    if constexpr (ALIGNED_FLAG == 1) {
                        quant_data = *reinterpret_cast<const int32_t *>(&global_mem[dim0 * ld + dim1].qs[
                            block_inter_index * 4]);
                    } else {
                        load_vec<int8_t, int32_t>(&global_mem[dim0 * ld + dim1].qs[block_inter_index * 4], &quant_data,
                                                  4);
                    }

                    quant_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K / sizeof(int) + PAD_SIZE) +
                             thread_x +
                             kk * (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK] = quant_data;
                } else {
                    quant_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K / sizeof(int) + PAD_SIZE) +
                             thread_x +
                             kk * (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK] = 0;
                }
            }
        }
        //load scale
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            int dim0 = start_block + warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD;
#pragma unroll
            for (int kk = 0; kk < VEC_DIM_K; ++kk) {
                int dim1 = k + quant_block_id + kk * QUANT_BLKS_PER_WARP / VEC_DIM_K;
                if (dim1 < ld && dim0 < dim) {
                    const tff::core::quant::Q_8_0 *val = global_mem + dim0 * ld + dim1;
                    scale_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (QUANT_BLKS_PER_WARP + PAD_SIZE) +
                             quant_block_id +
                             kk * QUANT_BLKS_PER_WARP / VEC_DIM_K] = val->d;
                } else {
                    scale_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (QUANT_BLKS_PER_WARP + PAD_SIZE) +
                             quant_block_id +
                             kk * QUANT_BLKS_PER_WARP / VEC_DIM_K] = 0;
                }
            }
        }
    }

    template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
        const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE, const int ALIGNED_FLAG>
    __device__ void load_tile_double_buffer(const int ld, const int dim,
                                            const int thread_x, const int warp_id,
                                            const int start_block,
                                            const int k, const int kk,
                                            const tff::core::quant::Q_8_0_ALIGNED *__restrict__ global_mem,
                                            int *quant_sm, half *scale_sm) {
        const int quant_block_id = thread_x / THREAD_NUM_PER_QUANT_BLOCK;
        const int block_inter_index = thread_x % THREAD_NUM_PER_QUANT_BLOCK;
        const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
        const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
        const int row_stride = BLOCK_DIM_LD / VEC_DIM_LD;
        const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
        const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;

        const int dim0_base = start_block + warp_id;
        //load quant data;
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            int dim0 = dim0_base + j * row_stride;

            int dim1 = k + quant_block_id;
            if (dim1 < ld && dim0 < dim) {
                int32_t quant_data = 0;
                if constexpr (ALIGNED_FLAG == 1) {
                    quant_data = *reinterpret_cast<const int32_t *>(&global_mem[dim0 * ld + dim1].qs[
                        block_inter_index * 4]);
                } else {
                    load_vec<int8_t, int32_t>(&global_mem[dim0 * ld + dim1].qs[block_inter_index * 4], &quant_data,
                                              4);
                }
                quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
                         kk * quant_col_stride] = quant_data;
            } else {
                quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
                         kk * quant_col_stride] = 0;
            }
        }

        //load scale
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            int dim0 = dim0_base + j * row_stride;

            int dim1 = k + quant_block_id;
            if (dim1 < ld && dim0 < dim) {
                const tff::core::quant::Q_8_0_ALIGNED *val = global_mem + dim0 * ld + dim1;
                scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
                         kk * scale_col_stride] = val->d;
            } else {
                scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
                         kk * scale_col_stride] = 0;
            }
        }
    }

    static __device__ int vec_dot_product(int a, int b, int c_sum) {
        return __dp4a(a, b, c_sum);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
        const int QUANT_BLKS_PER_WARP, const int PAD_SIZE,
        const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
    __device__ void compute_tile(const int thread_x, const int warp_id,
                                 const int *quant_a_sm, half *a_scale_sm,
                                 const int *quant_b_sm, half *b_scale_sm,
                                 float *c_reg) {
        const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
        const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
        const int m_row_stride = BLOCK_DIM_M / VEC_DIM_M;
        const int n_row_stride = BLOCK_DIM_N / VEC_DIM_N;
        const int4 *quant_a_sm_ptr = reinterpret_cast<const int4 *>(quant_a_sm);
        const int4 *quant_b_sm_ptr = reinterpret_cast<const int4 *>(quant_b_sm);

        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                float sum = 0;
#pragma unroll
                for (int kk = 0; kk < BLOCK_DIM_K / QUANT_BLOCK_SIZE; kk++) {
                    float a_scale[VEC_DIM_M];
                    float b_scale[VEC_DIM_N];
                    a_scale[mm] = __half2float(
                        a_scale_sm[(warp_id + mm * m_row_stride) * scale_col_width + kk]);
                    b_scale[nn] = __half2float(
                        b_scale_sm[(thread_x + nn * n_row_stride) * scale_col_width + kk]);
                    int block_sum = 0;
#pragma unroll
                    for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q += 4) {
                        const int kk_index = kk * VEC_DOT_PRODUCT + kk_q;
                        int4 a_reg[VEC_DIM_M] = {make_int4(0, 0, 0, 0)};
                        int4 b_reg[VEC_DIM_N] = {make_int4(0, 0, 0, 0)};
                        a_reg[mm] = quant_a_sm_ptr[(warp_id + mm * m_row_stride) * quant_col_width / 4 + kk_index / 4];
                        b_reg[nn] = quant_b_sm_ptr[(thread_x + nn * n_row_stride) * quant_col_width / 4 + kk_index / 4];

                        block_sum = vec_dot_product(a_reg[mm].x, b_reg[nn].x, block_sum);
                        block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
                        block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
                        block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
                    }
                    sum += (block_sum) * (a_scale[mm]) * (b_scale[nn]);
                }

                c_reg[mm * VEC_DIM_N + nn] += sum;
            }
        }
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
        const int QUANT_BLKS_PER_WARP, const int PAD_SIZE,
        const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT, const int THREAD_NUM_PER_QUANT_BLOCK>
    __device__ void compute_tile_double_buffer(const int k_size, const int thread_x, const int warp_id,
                                               const int k_block_index,
                                               const int *quant_a_sm, half *a_scale_sm,
                                               const int *quant_b_sm, half *b_scale_sm,
                                               float *c_reg) {
        const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
        const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
        const int m_row_stride = BLOCK_DIM_M / VEC_DIM_M;
        const int n_row_stride = BLOCK_DIM_N / VEC_DIM_N;
        const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
        const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
        const int4 *quant_a_sm_ptr = reinterpret_cast<const int4 *>(quant_a_sm);
        const int4 *quant_b_sm_ptr = reinterpret_cast<const int4 *>(quant_b_sm);

        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                float sum = 0;
#pragma unroll
                for (int kk = 0; kk < k_size; kk++) {
                    float a_scale[VEC_DIM_M];
                    float b_scale[VEC_DIM_N];
                    a_scale[mm] = __half2float(
                        a_scale_sm[(warp_id + mm * m_row_stride) * scale_col_width + kk +
                                   k_block_index * scale_col_stride]);
                    b_scale[nn] = __half2float(
                        b_scale_sm[(thread_x + nn * n_row_stride) * scale_col_width + kk +
                                   k_block_index * scale_col_stride]);
                    int block_sum = 0;
#pragma unroll
                    for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q += 4) {
                        const int kk_index = kk * VEC_DOT_PRODUCT + kk_q + k_block_index * quant_col_stride;
                        int4 a_reg[VEC_DIM_M] = {make_int4(0, 0, 0, 0)};
                        int4 b_reg[VEC_DIM_N] = {make_int4(0, 0, 0, 0)};
                        a_reg[mm] = quant_a_sm_ptr[(warp_id + mm * m_row_stride) * quant_col_width / 4 + kk_index / 4];
                        b_reg[nn] = quant_b_sm_ptr[(thread_x + nn * n_row_stride) * quant_col_width / 4 + kk_index / 4];

                        block_sum = vec_dot_product(a_reg[mm].x, b_reg[nn].x, block_sum);
                        block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
                        block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
                        block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
                    }

                    sum += (block_sum) * (a_scale[mm]) * (b_scale[nn]);
                }

                c_reg[mm * VEC_DIM_N + nn] += sum;
            }
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
        const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
        const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT, const int ALIGNED_FLAG>
    __global__ void mat_mul_quant_q_8_0(
        int M, int N, int K,
        int a_ld, int b_ld, int c_ld,
        const tff::core::quant::Q_8_0 *__restrict__ a,
        const tff::core::quant::Q_8_0 *__restrict__ b,
        T *__restrict__ c) {
        const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;

        const int warp_id = g_thread_id / (32);
        const int thread_x = g_thread_id % (32);

        const int start_m = blockIdx.y * BLOCK_DIM_M;
        const int start_n = blockIdx.x * BLOCK_DIM_N;

        __shared__ int a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
        __shared__ int b_quant_data_sm[BLOCK_DIM_N][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
        __shared__ half a_scale_data_sm[BLOCK_DIM_M][QUANT_BLKS_PER_WARP + PAD_SIZE];
        __shared__ half b_scale_data_sm[BLOCK_DIM_N][QUANT_BLKS_PER_WARP + PAD_SIZE];

        float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};

        for (size_t k = 0; k < K; k += BLOCK_DIM_K / QUANT_BLOCK_SIZE) {
            load_tile<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK,
                PAD_SIZE, ALIGNED_FLAG>(
                a_ld, M, thread_x, warp_id, start_m, k, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
            load_tile<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK,
                PAD_SIZE, ALIGNED_FLAG>(
                b_ld, N, thread_x, warp_id, start_n, k, b, &b_quant_data_sm[0][0], &b_scale_data_sm[0][0]);
            __syncthreads();

            compute_tile<VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP, PAD_SIZE,
                QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT>(
                thread_x, warp_id,
                &a_quant_data_sm[0][0], &a_scale_data_sm[0][0],
                &b_quant_data_sm[0][0], &b_scale_data_sm[0][0],
                &c_reg[0][0]);

            __syncthreads();
        }

        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            const int dst_m_index = start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
            if (dst_m_index >= M) {
                continue;
            }
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                const int dst_n_index = start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N;
                if (dst_n_index >= N) {
                    continue;
                }
                if (std::is_same_v<T, half>) {
                    half old_value = c[dst_m_index * c_ld + dst_n_index];
                    c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
                } else if (std::is_same_v<T, float>) {
                    c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
                }
            }
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
        const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
        const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT, const int ALIGNED_FLAG>
    __global__ void mat_mul_quant_q_8_0_double_buffer(
        int M, int N, int K,
        int a_ld, int b_ld, int c_ld,
        const tff::core::quant::Q_8_0_ALIGNED *__restrict__ a,
        const tff::core::quant::Q_8_0_ALIGNED *__restrict__ b,
        T *__restrict__ c) {
        const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;

        const int warp_id = g_thread_id / (32);
        const int thread_x = g_thread_id % (32);

        const int start_m = blockIdx.y * BLOCK_DIM_M;
        const int start_n = blockIdx.x * BLOCK_DIM_N;

        __shared__ int a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
        __shared__ int b_quant_data_sm[BLOCK_DIM_N][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
        __shared__ half a_scale_data_sm[BLOCK_DIM_M][QUANT_BLKS_PER_WARP + PAD_SIZE];
        __shared__ half b_scale_data_sm[BLOCK_DIM_N][QUANT_BLKS_PER_WARP + PAD_SIZE];

        float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};
        int flip_flag = 0;

        load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
            THREAD_NUM_PER_QUANT_BLOCK,
            PAD_SIZE, ALIGNED_FLAG>(
            a_ld, M, thread_x, warp_id, start_m, 0, flip_flag, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
        load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
            THREAD_NUM_PER_QUANT_BLOCK,
            PAD_SIZE, ALIGNED_FLAG>(
            b_ld, N, thread_x, warp_id, start_n, 0, flip_flag, b, &b_quant_data_sm[0][0], &b_scale_data_sm[0][0]);
        __syncthreads();

        for (int k = 0; k <= K; k += ((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K)) {
            const int k_size = min((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K, K - k);
            compute_tile_double_buffer<VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
                QUANT_BLKS_PER_WARP,
                PAD_SIZE,
                QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT, THREAD_NUM_PER_QUANT_BLOCK>(
                k_size, thread_x, warp_id, flip_flag,
                &a_quant_data_sm[0][0], &a_scale_data_sm[0][0],
                &b_quant_data_sm[0][0], &b_scale_data_sm[0][0],
                &c_reg[0][0]);

            const int next_k = k + (BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K;
            if (next_k < K) {
                //printf("error:  next_k: %d,flip_flag: %d !flip_flag: %d \n", next_k, flip_flag, !flip_flag);
                load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
                    THREAD_NUM_PER_QUANT_BLOCK,
                    PAD_SIZE, ALIGNED_FLAG>(
                    a_ld, M, thread_x, warp_id, start_m, next_k, !flip_flag, a, &a_quant_data_sm[0][0],
                    &a_scale_data_sm[0][0]);
                load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
                    THREAD_NUM_PER_QUANT_BLOCK,
                    PAD_SIZE, ALIGNED_FLAG>(
                    b_ld, N, thread_x, warp_id, start_n, next_k, !flip_flag, b, &b_quant_data_sm[0][0],
                    &b_scale_data_sm[0][0]);
            }
            __syncthreads();
            flip_flag ^= 1;
        }

        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            const int dst_m_index = start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
            if (dst_m_index >= M) {
                continue;
            }
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                const int dst_n_index = start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N;
                if (dst_n_index >= N) {
                    continue;
                }
                if (std::is_same_v<T, half>) {
                    half old_value = c[dst_m_index * c_ld + dst_n_index];
                    c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
                } else if (std::is_same_v<T, float>) {
                    c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
                }
            }
        }
    }

    template<typename T>
    void quant_q_8_0_matmul(const int M, const int N, const int K,
                            std::shared_ptr<tff::core::memory::Tensor> &quant_a,
                            std::shared_ptr<tff::core::memory::Tensor> &quant_b,
                            std::shared_ptr<tff::core::memory::Tensor> &c,
                            std::shared_ptr<core::device::DeviceStream> &stream) {
        constexpr int QUANT_BLOCK_SIZE = tff::core::quant::Q_8_0::BLOCK_SIZE;
        constexpr int VEC_DIM_M = 8;
        constexpr int VEC_DIM_N = 2;
        constexpr int VEC_DIM_K = 2;
        constexpr int BLOCK_M_DIM = 64;
        constexpr int BLOCK_N_DIM = 64;
        constexpr int BLOCK_K_DIM = 256;
        constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
        constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int);
        constexpr int PAD_SIZE = 4;
        constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);

        dim3 grid((N + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (M + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
        dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
        mat_mul_quant_q_8_0_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM,
            BLOCK_K_DIM,
            QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, PAD_SIZE, QUANT_BLOCK_SIZE,
            VEC_DOT_PRODUCT, 1><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>>(
            M, N, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, N,
            static_cast<tff::core::quant::Q_8_0_ALIGNED *>(quant_a->get_buffer()->ptr()),
            static_cast<tff::core::quant::Q_8_0_ALIGNED *>(quant_b->get_buffer()->ptr()),
            static_cast<T *>(c->get_buffer()->ptr()));
    }

#ifdef _DEBUG
    static void varify(std::string &filename, std::shared_ptr<core::memory::Tensor> &tensor) {
        switch (tensor->get_data_type()) {
            case core::memory::DataType::TFF_DATA_TYPE_F32: {
                std::vector<float> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<float> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                    for (int nn = 0; nn < tensor->get_shape()[0]; nn++) {
                        float delta = weight_gpu_result[mm * tensor->get_shape()[0] + nn] - weight_cpu_result[
                                          mm * tensor->get_shape()[0] + nn];
                        if (fabs(delta) > 0.001f) {
                            tff::log::Logger::error("filename: %s, error: m: %d n: %d, delta: %lf", filename.c_str(), nn, delta);
                            throw std::runtime_error("error");
                        }
                    }
                }
                break;
            }
            case core::memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED: {
                std::vector<Q8_0> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] / Q8_0::BLOCK_SIZE * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<Q8_0_ALIGNED> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                    for (int nn = 0; nn < tensor->get_shape()[0]/Q8_0::BLOCK_SIZE; nn++) {
                        float delta = weight_gpu_result[mm * tensor->get_shape()[0] /Q8_0::BLOCK_SIZE  + nn].d -
                            __half2float(weight_cpu_result[
                                          mm * tensor->get_shape()[0]/Q8_0::BLOCK_SIZE + nn].d);
                        if (fabs(delta) > 0.001f) {
                            tff::log::Logger::error("filename: %s, error: m: %d n: %d, delta: %lf", filename.c_str(),mm, nn, delta);
                            throw std::runtime_error("error");
                        }
                    }
                }
                break;
            }
            default:
                break;
        }

        tff::log::Logger::info("layer node op varify (%s) success!", filename.c_str());
    }
#endif
    template<typename T>
    void tff::kernel::QuantQ8MatMul<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {

        auto weight_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        auto x_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            3, para_ptr);
        auto mem_buffer_manager_ptr = kernel::base::get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(4, para_ptr);
        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        para_ptr->get_param_count() - 1, para_ptr);

        if (weight_tensor == nullptr || x_tensor == nullptr || output_tensor == nullptr) {
            return;
        }

        if (weight_tensor->get_buffer() == nullptr || weight_tensor->get_buffer()->ptr() == nullptr) {
            tff::log::Logger::error("weight tensor is null!");
            return;
        }
        if (x_tensor->get_buffer() == nullptr || x_tensor->get_buffer()->ptr() == nullptr) {
            tff::log::Logger::error("x tensor is null!");
            return;
        }
        const int K = weight_tensor->get_shape()[0];
        const int M = weight_tensor->get_shape()[1];
        const int B = weight_tensor->get_shape()[2]; //todo impl batches
        const int N = x_tensor->get_shape()[1];

        //
        quant_q_8_0_matmul<T>(M, N, K, weight_tensor,
                              x_tensor, output_tensor,
                              stream);

#ifdef _DEBUG
        const auto &name = kernel::base::get_param_value<std::string>(para_ptr->get_param_count() - 5, para_ptr);
        std::string filename = "";
        if (name == "blk.0.attn_q_mul_w") {
            filename = "Qcur-0_src_0.ggml";
        } else if (name == "blk.0.attn_k_mul_w") {
            filename = "Kcur-0_src_0.ggml";
        } else if (name == "blk.0.attn_v_mul_w") {
            filename = "Vcur-0_src_0.ggml";
        }
        varify(filename, weight_tensor);
        if (name == "blk.0.attn_q_mul_w") {

            filename = "Qcur-0_src_1.ggml";
        } else if (name == "blk.0.attn_k_mul_w") {
            filename = "Kcur-0_src_1.ggml";
        } else if (name == "blk.0.attn_v_mul_w") {
            filename = "Vcur-0_src_1.ggml";
        }
        //varify(filename, x_tensor);

        if (name == "blk.0.attn_q_mul_w") {
            filename = "Qcur-0_result.ggml";
        } else if (name == "blk.0.attn_k_mul_w") {
            filename = "Kcur-0_result.ggml";
        } else if (name == "blk.0.attn_v_mul_w") {
            filename = "Vcur-0_result.ggml";
        }
        varify(filename, output_tensor);
#endif
    }

    template class tff::kernel::QuantQ8MatMul<float>;
    template class tff::kernel::QuantQ8MatMul<half>;
    REGISTER_OP_OBJECT(QuantQ8MatMul, float);

    REGISTER_OP_OBJECT(QuantQ8MatMul, half);
}
