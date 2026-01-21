//
// Created by nkk on 2026/1/3.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
        const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE>
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
                    const tff::core::quant::Q_8_0 *val = &global_mem[dim0 * ld + dim1];
                    const int *quant_data = reinterpret_cast<const int *>(&val->qs[0]);
                    quant_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K / sizeof(int) + PAD_SIZE) +
                             thread_x +
                             kk * (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK] = quant_data[
                        block_inter_index];
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
        const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE>
    __device__ void load_tile_double_buffer(const int ld, const int dim,
                                            const int thread_x, const int warp_id,
                                            const int start_block,
                                            const int k, const int kk,
                                            const tff::core::quant::Q_8_0 *__restrict__ global_mem,
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
                const tff::core::quant::Q_8_0 *val = &global_mem[dim0 * ld + dim1];
                const int *quant_data = reinterpret_cast<const int *>(&val->qs[0]);
                quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
                         kk * quant_col_stride] = quant_data[block_inter_index];
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
                const tff::core::quant::Q_8_0 *val = global_mem + dim0 * ld + dim1;
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
        const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
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
                PAD_SIZE>(
                a_ld, M, thread_x, warp_id, start_m, k, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
            load_tile<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK,
                PAD_SIZE>(
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
        const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
    __global__ void mat_mul_quant_q_8_0_double_buffer(
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
        int flip_flag = 0;

        load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
            THREAD_NUM_PER_QUANT_BLOCK,
            PAD_SIZE>(
            a_ld, M, thread_x, warp_id, start_m, 0, flip_flag, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
        load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
            THREAD_NUM_PER_QUANT_BLOCK,
            PAD_SIZE>(
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
                    PAD_SIZE>(
                    a_ld, M, thread_x, warp_id, start_m, next_k, !flip_flag, a, &a_quant_data_sm[0][0],
                    &a_scale_data_sm[0][0]);
                load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
                    THREAD_NUM_PER_QUANT_BLOCK,
                    PAD_SIZE>(
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
                            void *quant_a,
                            void *quant_b,
                            T *c) {
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
        if (std::max(M, N) > 2048) {
            mat_mul_quant_q_8_0<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM, BLOCK_K_DIM,
                        QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, PAD_SIZE, QUANT_BLOCK_SIZE,
                        VEC_DOT_PRODUCT><<<grid
                    , block>>>(
                        M, N, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, N,
                        static_cast<tff::core::quant::Q_8_0 *>(quant_a),
                        static_cast<tff::core::quant::Q_8_0 *>(quant_b),
                        c);
        } else {
            mat_mul_quant_q_8_0_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM,
                        BLOCK_K_DIM,
                        QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, PAD_SIZE, QUANT_BLOCK_SIZE,
                        VEC_DOT_PRODUCT><<<grid
                    , block>>>(
                        M, N, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, N,
                        static_cast<tff::core::quant::Q_8_0 *>(quant_a),
                        static_cast<tff::core::quant::Q_8_0 *>(quant_b),
                        c);
        }
    }

    template<typename T>
    void tff::kernel::QuantQ8MatMul<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), QuantQ8MatMul<T>::get_op_name().c_str());
        auto weight_tensor = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        auto x_tensor = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            3, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(4, para_ptr);

        if (weight_tensor == nullptr || x_tensor == nullptr || output_tensors == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }

        if (weight_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("weight tensor is null!");
            return;
        }
        if (x_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("x tensor is null!");
            return;
        }
        const int K = weight_tensor->get_shape()[0];
        const int M = weight_tensor->get_shape()[1];
        const int B = weight_tensor->get_shape()[2]; //todo impl batches
        const int N = x_tensor->get_shape()[1];

        //
        quant_q_8_0_matmul<T>(M, N, K, weight_tensor->get_buffer()->ptr(),
                              x_tensor->get_buffer()->ptr(), static_cast<T *>(output_tensors->get_buffer()->ptr()));

        // mem_buffer_manager_ptr->reset_gpu_memory(weight_tensor->get_external_memory_index());
        // mem_buffer_manager_ptr->reset_gpu_memory(x_tensor->get_external_memory_index());// todo 需要区分全部加载权重和边推理边加载权重
    }

    template<typename T>
    std::string tff::kernel::QuantQ8MatMul<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_MATMUL);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();

        return name;
    }

    template class tff::kernel::QuantQ8MatMul<float>;
    template class tff::kernel::QuantQ8MatMul<half>;
    REGISTER_OP_OBJECT(QuantQ8MatMul, float);

    REGISTER_OP_OBJECT(QuantQ8MatMul, half);
}
