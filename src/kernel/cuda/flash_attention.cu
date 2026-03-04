//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#define _PTX
#define _ASYNC
#define _DOUBER_BUFFER

namespace tff::kernel {
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

    static __device__ __forceinline__ void cp_async_wait_all() {
        asm volatile("cp.async.wait_all;");
    }

    static __device__ __forceinline__ void cp_async_commit() {
        asm volatile("cp.async.commit_group;");
    }

    template<int preload>
    static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void *src) {
        if (preload == 256) {
            asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
                : : "r"(dst), "l"(src));
        } else if (preload == 128) {
            asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                : : "r"(dst), "l"(src));
        } else if (preload == 64) {
            asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
                : : "r"(dst), "l"(src));
        } else {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                : : "r"(dst), "l"(src));
        }
    }

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

    template<typename T1, typename T2>
    __device__ __forceinline__ void load_vec_async(const T1 *addr, T2 *out);

    template<>
    __device__ __forceinline__ void load_vec_async<half, half2>(const half *addr, half2 *out) {
        if (reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
#ifndef _PTX
            const half2 *h = reinterpret_cast<const half2 *>(addr);
            out[0] = h[0];
#else
            const unsigned int dst_addr = __cvta_generic_to_shared(out);
            cp_async_cg_16<64>(dst_addr, addr);
            cp_async_commit();
#endif
        }
    }

    template<>
    __device__ __forceinline__ void load_vec_async<float, float2>(const float *addr, float2 *out) {
        if (reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
#ifndef _PTX
            const float2 *h = reinterpret_cast<const float2 *>(addr);
            out[0] = h[0];
#else
            const unsigned int dst_addr = __cvta_generic_to_shared(out);
            cp_async_cg_16<64>(dst_addr, addr);
            cp_async_commit();
#endif
        }
    }

    template<>
    __device__ __forceinline__ void load_vec_async<float, half2>(const float *addr, half2 *out) {
        if (reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
#ifndef _PTX
            const float2 *h = reinterpret_cast<const float2 *>(addr);
            out[0] = __float22half2_rn(h[0]);
#else
            const unsigned int dst_addr = __cvta_generic_to_shared(out);
            cp_async_cg_16<64>(dst_addr, addr);
            cp_async_commit();
#endif
        }
    }

    template<const int ELEMENTS_PER_LOAD>
    __device__ __forceinline__ void rote(const int addr, half2 *sm, float2 *cos_sin_table) {
        if constexpr (ELEMENTS_PER_LOAD == 8) {
            cp_async_wait_all();
            auto val_vec = reinterpret_cast<float4 *>(&sm[addr]);
            float4 sm_val = val_vec[0];
            half2 *rot_value = reinterpret_cast<half2 *>(&sm_val);
#pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                half2 val = rot_value[i];
                rot_value[i] = complex_mul_half2(val, __float22half2_rn(cos_sin_table[i]));
            }
            float4 k_rot = *reinterpret_cast<float4 *>(&rot_value[0]);
            val_vec[0] = k_rot;
        } else if constexpr (ELEMENTS_PER_LOAD == 2) {
#pragma unroll
            for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                sm[addr + i] = complex_mul_half2(sm[addr + i], __float22half2_rn(cos_sin_table[i]));
            }
        }
    }

    template<const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const
        int
        S, const int PAD_SIZE>
    __device__ __forceinline__ void rope_k(const int ld, const int dim,
                                           const int thread_x, const int warp_id,
                                           const int start_m,
                                           const int k,
                                           const float *__restrict__ cos_sin_table,
                                           half2 *k_sm) {
        //rot;
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                const int dim1 = k + thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                                 ELEMENTS_PER_LOAD;

                float2 cos_sin_table_val[ELEMENTS_PER_LOAD / 2];
                if (dim0_base < dim) {
                    const int actual_load = min(ELEMENTS_PER_LOAD / 2, ld - dim1);
                    if (actual_load > 0) {
                        load_vec<float, float2>(&cos_sin_table[dim0_base * ld + dim1], &cos_sin_table_val[0],
                                                actual_load);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                        cos_sin_table_val[i].x = 0;
                        cos_sin_table_val[i].y = 0;
                    }
                }

                int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
                int sm_col = thread_x * ELEMENTS_PER_LOAD / 2 + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD / 2;
                int offset = sm_row * (BLOCK_DIM_K / 2 + PAD_SIZE) + sm_col;
#ifdef _SWIZZLE
                int addr8 = swizzle<B, MBit, S>(offset);
#else
                int addr8 = offset; //
#endif
                //
                rote<ELEMENTS_PER_LOAD>(addr8, k_sm, cos_sin_table_val);
            }
        }
    }

    template<const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const
        int
        S, const int PAD_SIZE, const int HEAD_NUM_PER_BLOCK, const int HIDDEN_DIM>
    __device__ __forceinline__ void rope_q(const int ld, const int dim,
                                           const int thread_x, const int warp_id,
                                           const int start_m,
                                           const int k,
                                           const float *__restrict__ cos_sin_table,
                                           half2 *q_sm) {
        //rot;
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);

            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                const int dim1 = thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                                 ELEMENTS_PER_LOAD;

                float2 cos_sin_table_val[ELEMENTS_PER_LOAD / 2];
                if (dim0_base < dim) {
                    const int actual_load = min(ELEMENTS_PER_LOAD / 2, ld - dim1);
                    if (actual_load > 0) {
                        load_vec<float, float2>(&cos_sin_table[dim0_base * ld + dim1], &cos_sin_table_val[0],
                                                actual_load);
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                        cos_sin_table_val[i].x = 0;
                        cos_sin_table_val[i].y = 0;
                    }
                }
#pragma unroll
                for (int head_index = 0; head_index < HEAD_NUM_PER_BLOCK; ++head_index) {
                    int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
                    int sm_col = head_index * HIDDEN_DIM / 2 + thread_x * ELEMENTS_PER_LOAD / 2 +
                                 kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD / 2;
                    int offset = sm_row * (HEAD_NUM_PER_BLOCK * HIDDEN_DIM / 2 + PAD_SIZE) + sm_col;
#ifdef _SWIZZLE
                    int addr = swizzle<B, MBit, S>(offset);
#else
                    int addr = offset; //
#endif
#pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                        q_sm[addr + i] = complex_mul_half2(q_sm[addr + i], __float22half2_rn(cos_sin_table_val[i]));
                    }
                }
            }
        }
    }

    template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const
        int
        S, const int PAD_SIZE>
    __device__ __forceinline__ void load_tile_q(const int ld, const int dim,
                                                const int thread_x, const int warp_id,
                                                const int start_m,
                                                const int k,
                                                const T *__restrict__ global_mem,
                                                half2 *sm) {
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                const int dim1 = k + thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                                 ELEMENTS_PER_LOAD;

                int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
                int sm_col = thread_x * ELEMENTS_PER_LOAD / 2 + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD / 2;
                int offset = sm_row * (BLOCK_DIM_K / 2 + PAD_SIZE) + sm_col;
#ifdef _SWIZZLE
                int addr8 = swizzle<B, MBit, S>(offset);
#else
                int addr8 = offset; //
#endif
#ifndef _ASYNC
                half2 val[ELEMENTS_PER_LOAD / 2];
#endif
                if (dim0_base < dim) {
                    const int actual_load = min(ELEMENTS_PER_LOAD, ld - dim1);
                    if (actual_load > 0) {
#ifdef _ASYNC
                        load_vec_async<T, half2>(&global_mem[dim0_base * ld + dim1], &sm[addr8]);
#else
                        load_vec<T, half2>(&global_mem[dim0_base * ld + dim1], &val[0], ELEMENTS_PER_LOAD / 2);
                        sm[addr8] = val[0];
#endif
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                        int addr = swizzle<B, MBit, S>(offset + i);
                        sm[addr].x = 0;
                        sm[addr].y = 0;
                    }
                }
            }
        }
    }

    template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const
        int
        S, const int PAD_SIZE>
    __device__ __forceinline__ void load_tile_kv(const int ld, const int dim,
                                                 const int thread_x, const int warp_id,
                                                 const int start_m,
                                                 const int k,
                                                 const T *__restrict__ global_mem,
                                                 half2 *sm) {
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                const int dim1 = k + thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
                                 ELEMENTS_PER_LOAD;

                int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
                int sm_col = thread_x * ELEMENTS_PER_LOAD / 2 + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD / 2;
                int offset = sm_row * (BLOCK_DIM_K / 2 + PAD_SIZE) + sm_col;
#ifdef _SWIZZLE
                int addr8 = swizzle<B, MBit, S>(offset);
#else
                int addr8 = offset; //
#endif

                half2 val[ELEMENTS_PER_LOAD / 2];
                if (dim0_base < dim) {
                    const int actual_load = min(ELEMENTS_PER_LOAD / 2, ld - dim1);
                    if (actual_load > 0) {
                        load_vec<T, half2>(&global_mem[dim0_base * ld + dim1], &val[0], ELEMENTS_PER_LOAD / 2);
                        sm[addr8] = val[0];
                    }
                } else {
#pragma unroll
                    for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                        int addr = swizzle<B, MBit, S>(offset + i);
                        sm[addr].x = 0;
                        sm[addr].y = 0;
                    }
                }
            }
        }
    }


    template<typename T_KV, const int VEC_DIM_M, const int VEC_DIM_N,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int HIDDEN_DIM,
        const int HEAD_NUM_PER_BLOCK, const int B, const int MBit, const int S,
        const int PAD_SIZE_Q, const int PAD_SIZE_KV>
    __device__ __forceinline__ void compute_tile_attention_gemm(const int N, const int k_size,
                                                                const int thread_x,
                                                                const int warp_id,
                                                                const int start_m, const int start_n,
                                                                half2 *q_sm,
                                                                half2 *b_sm,
                                                                const T_KV *mask,
                                                                float *c_reg) {
        const int reg_stride = VEC_DIM_N * HEAD_NUM_PER_BLOCK;
        const int n_stride = BLOCK_DIM_N / VEC_DIM_N;
        const int m_stride = BLOCK_DIM_M / VEC_DIM_M;

#pragma unroll
        for (int head_index = 0; head_index < HEAD_NUM_PER_BLOCK; head_index++) {
            for (int kk = head_index * HIDDEN_DIM / 2; kk < (head_index + 1) * HIDDEN_DIM / 2; kk++) {
                const int stride = (BLOCK_DIM_K / 2 + PAD_SIZE_Q);

#pragma unroll
                for (int mm = 0; mm < VEC_DIM_M; mm++) {
                    int m_row_base = (warp_id + mm * m_stride);
                    int col_base = kk;
                    int q_index = m_row_base * stride + col_base;
#ifdef _SWIZZLE
                    int q_addr = swizzle<B, MBit, S>(q_index);
#else
                    int q_addr = q_index;
#endif
                    float2 q_rot = __half22float2(q_sm[q_addr]);

#pragma unroll
                    for (int nn = 0; nn < VEC_DIM_N; nn++) {
                        int n_row_base = (thread_x + nn * n_stride);
                        int k_index = col_base % (HIDDEN_DIM / 2) + n_row_base * (HIDDEN_DIM / 2 + PAD_SIZE_KV);

#ifdef _SWIZZLE
                        int k_addr = swizzle<B, MBit, S>(k_index);
#else
                        int k_addr = k_index;
#endif
                        float2 k_rot = __half22float2(b_sm[k_addr]);

                        const float mask_value = __ldg(&mask[(start_m + m_row_base) * N + start_n + n_row_base]);

                        c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N] += fmaf(
                            q_rot.x, k_rot.x, fmaf(q_rot.y, k_rot.y,
                                                   mask_value));
                    }
                }
            }
        }
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int
        BLOCK_DIM_K, const int HIDDEN_DIM, const int HEAD_NUM_PER_BLOCK, const int B, const int MBit, const int S
        , const int PAD_SIZE_Q, const int PAD_SIZE_KV>
    __device__ __forceinline__ void compute_softmax_pv(const int N, const int v_ld, const int start_m,
                                                       const int start_n,
                                                       const int start_d,
                                                       const int thread_x, const int warp_id,
                                                       float *max_value, float *sum_value, half2 *sm,
                                                       const float scale,
                                                       float *c_reg,
                                                       half2 *output_sm) {
        const int base_n = start_n + thread_x;
        const int valid_n = min(VEC_DIM_N, (N - base_n + 31) / 32);
        const int stride = (BLOCK_DIM_K / 2 + PAD_SIZE_Q);
        const int n_stride = BLOCK_DIM_N / VEC_DIM_N;
        const int m_stride = BLOCK_DIM_M / VEC_DIM_M;
        const int reg_stride = VEC_DIM_N * HEAD_NUM_PER_BLOCK;

#pragma unroll
        for (int head_index = 0; head_index < HEAD_NUM_PER_BLOCK; head_index++) {
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                const float old_max = max_value[mm * HEAD_NUM_PER_BLOCK + head_index];
                const float old_sum = sum_value[mm * HEAD_NUM_PER_BLOCK + head_index];
                float new_max_value = {-1e20f};
                float new_sum_value = {0.0f};
                float current_max_value = {-1e20f};
                float current_sum_value = {0.0f};

                for (int nn = 0; nn < valid_n; nn++) {
                    c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N] *= scale;
                    current_max_value = fmaxf(current_max_value,
                                              c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N]);
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    current_max_value = fmaxf(current_max_value,
                                              __shfl_xor_sync(0xffffffff, current_max_value, offset,
                                                              32));
                }

                //
                for (int nn = 0; nn < valid_n; nn++) {
                    float c_reg_value = c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N] - current_max_value;
                    c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N] = expf(c_reg_value);
                    current_sum_value += c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N];
                }

#pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    current_sum_value += __shfl_xor_sync(0xffffffff, current_sum_value, offset, 32);
                }

                new_max_value = fmaxf(current_max_value, old_max);
                float exp_old = expf(old_max - new_max_value);
                exp_old *= old_sum;
                float exp_local = expf(current_max_value - new_max_value);
                new_sum_value = current_sum_value * exp_local + exp_old;

                for (int kk = head_index * HIDDEN_DIM / 2; kk < (head_index + 1) * HIDDEN_DIM / 2; kk++) {
                    float2 acc;
                    acc.x = 0;
                    acc.y = 0;
                    for (int nn = 0; nn < valid_n; nn++) {
                        const int n_row_base = (thread_x + nn * n_stride);
                        int index = kk % (HIDDEN_DIM / 2) + n_row_base * (HIDDEN_DIM / 2 + PAD_SIZE_KV);
#ifdef _SWIZZLE
                        int addr = swizzle<B, MBit, S>(index);
#else
                        int addr = index;
#endif
                        float2 v_reg_0 = __half22float2(sm[addr]);

                        acc.x = fmaf(v_reg_0.x, c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N], acc.x);
                        acc.y = fmaf(v_reg_0.y, c_reg[mm * reg_stride + nn + head_index * VEC_DIM_N], acc.y);
                    }
#pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        acc.x += __shfl_xor_sync(0xffffffff, acc.x, offset, 32);
                        acc.y += __shfl_xor_sync(0xffffffff, acc.y, offset, 32);
                    }

                    int output_index = (warp_id + mm * m_stride) * stride + kk;
                    half2 output_val = output_sm[output_index];
                    output_val.x = __float2half(
                        (fmaf(__half2float(output_val.x), exp_old,
                              fmaf(acc.x, exp_local, 0.0f))) / new_sum_value);
                    output_val.y = __float2half(
                        (fmaf(__half2float(output_val.y), exp_old,
                              fmaf(acc.y, exp_local, 0.0f))) / new_sum_value);

                    output_sm[output_index] = output_val;
                }


                max_value[mm * HEAD_NUM_PER_BLOCK + head_index] = new_max_value;
                sum_value[mm * HEAD_NUM_PER_BLOCK + head_index] = new_sum_value;
            }
        }
    }

    template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
        const int ELEMENTS_PER_LOAD, const int PAD_SIZE>
    __device__ __forceinline__ void init(
        const int thread_x, const int warp_id,
        half2 *sm, const float value) {
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            //VEC_DIM_LD = 4;BLOCK_DIM_LD = 32;
            int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                //VEC_DIM_K = 2;BLOCK_DIM_K = 64
                int sm_col = thread_x + kk * (BLOCK_DIM_K / VEC_DIM_K);
                int offset = sm_row * (BLOCK_DIM_K / 2 + PAD_SIZE) + sm_col;
                sm[offset].x = __float2half(value);
                sm[offset].y = __float2half(value);
            }
        }
    }

    template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int
        ELEMENTS_PER_LOAD, const int PAD_SIZE>
    __device__ __forceinline__ void store(const int M,
                                          const int ld,
                                          const int start_m,
                                          const int warp_id, const int thread_x,
                                          half2 *output_sm,
                                          float *out_put) {
        float2 *output_ptr = reinterpret_cast<float2 *>(out_put);
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            int row_base = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
#pragma unroll
            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                int col_base = thread_x * ELEMENTS_PER_LOAD / 2 + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD /
                               2;
                for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
                    float2 out = __half22float2(output_sm[row_base * (BLOCK_DIM_K / 2 + PAD_SIZE) + col_base + i]);
                    if ((start_m + row_base) < M) {
                        output_ptr[(start_m + row_base) * ld / 2 + col_base + i] = out;
                    }
                }
            }
        }
    }

    template<typename T_Q, typename T_KV, const int NUM_Q_HEAD, const int NUM_KV_HEAD, const int HIDDEN_DIM,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N,
        const int ELEMENTS_PER_LOAD,
        const int rope_flag, const int B, const int MBit, const int S, const int HEAD_NUM_PER_BLOCK,
        const int WARP_SIZE, const int THREAD_BLOCK>
    __global__ void flash_attention_fp16_8x32(const int M, const int N, const int D,
                                              const int MAX_CTX,
                                              const int q_ld, const int k_ld, const int v_ld,
                                              const float scale,
                                              const T_Q *__restrict__ q_global,
                                              const T_KV *__restrict__ k_global,
                                              const T_KV *__restrict__ v_global,
                                              const T_KV *mask,
                                              float *__restrict__ cos_sin_table,
                                              float *out_put) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        const int thread_x = thread_id % WARP_SIZE;
        const int warp_id = thread_id / WARP_SIZE;
        const int head_index = blockIdx.y * HEAD_NUM_PER_BLOCK;
        const int kv_group_per_q = NUM_Q_HEAD / NUM_KV_HEAD;
        const int kv_group_start_index = head_index / kv_group_per_q;

        constexpr int BLOCK_DIM_K_KV = HIDDEN_DIM;
        constexpr int BLOCK_DIM_K_Q = HEAD_NUM_PER_BLOCK * HIDDEN_DIM;
        constexpr int VEC_DIM_M = BLOCK_DIM_M / (THREAD_BLOCK / WARP_SIZE);
        constexpr int VEC_DIM_N = BLOCK_DIM_N / (THREAD_BLOCK / WARP_SIZE);
        constexpr int VEC_DIM_N_CM = BLOCK_DIM_N / WARP_SIZE;
        constexpr int VEC_DIM_K_Q = BLOCK_DIM_K_Q / WARP_SIZE;
        constexpr int VEC_DIM_K_KV = BLOCK_DIM_K_KV / WARP_SIZE;
        constexpr int ELEMENTS_PER_LOAD_KV = 2;
        constexpr int PAD_SIZE_KV = 1;
        constexpr int PAD_SIZE_Q = 4;

        const int block_x = blockIdx.x;
        const int start_m = block_x * BLOCK_DIM_M;

        const int kv_stride = NUM_KV_HEAD * k_ld;
        const int q_stride = NUM_Q_HEAD * q_ld;

        const T_Q *q_ptr = q_global + (blockIdx.z * NUM_Q_HEAD) * M * D + head_index * D;
        const T_KV *k_ptr = k_global + (blockIdx.z * NUM_KV_HEAD) * N * D + kv_group_start_index * D;
        const T_KV *v_ptr = v_global + (blockIdx.z * NUM_KV_HEAD) * N * D + kv_group_start_index * D;
        float *output = out_put + (blockIdx.z * NUM_Q_HEAD) * M * D + head_index * D;

        constexpr int shared_mem_block_size = (BLOCK_DIM_K_KV / 2 + PAD_SIZE_KV) * (BLOCK_DIM_N);
        __shared__ half2 sm[shared_mem_block_size * 2]; //2 * kv;(k, v共享一块shared mem)


        constexpr int q_sm_size = (BLOCK_DIM_K_Q / 2 + PAD_SIZE_Q) * (BLOCK_DIM_M);
        __shared__ half2 q_sm[q_sm_size];

        constexpr int out_sm_size = (BLOCK_DIM_K_Q / 2 + PAD_SIZE_Q) * (BLOCK_DIM_M);
        __shared__ half2 out_sm[out_sm_size];
        init<VEC_DIM_M, VEC_DIM_K_Q, BLOCK_DIM_M, BLOCK_DIM_K_Q, ELEMENTS_PER_LOAD_KV, PAD_SIZE_Q>(
            thread_x, warp_id, &out_sm[0], 0);


        int flip_flag = 0;
        load_tile_kv<T_KV, VEC_DIM_N, VEC_DIM_K_KV, BLOCK_DIM_N, BLOCK_DIM_K_KV, ELEMENTS_PER_LOAD_KV, B, MBit, S,
            PAD_SIZE_KV>(
            kv_stride, N, thread_x, warp_id, 0, 0,
            k_ptr, &sm[(flip_flag) * shared_mem_block_size]);

        load_tile_q<T_Q, VEC_DIM_M, VEC_DIM_K_Q, BLOCK_DIM_M, BLOCK_DIM_K_Q, ELEMENTS_PER_LOAD, B, MBit, S, PAD_SIZE_Q>(
            q_stride, M, thread_x, warp_id, start_m, 0,
            q_ptr, &q_sm[0]);

        if constexpr (rope_flag == 1) {
            rope_k<VEC_DIM_N, VEC_DIM_K_KV, BLOCK_DIM_N, BLOCK_DIM_K_KV, ELEMENTS_PER_LOAD_KV, B, MBit, S, PAD_SIZE_KV>(
                k_ld, MAX_CTX, thread_x, warp_id, 0, 0,
                cos_sin_table, &sm[(flip_flag) * shared_mem_block_size]);

            rope_q<VEC_DIM_M, VEC_DIM_K_KV, BLOCK_DIM_M, BLOCK_DIM_K_KV, ELEMENTS_PER_LOAD_KV, B, MBit, S, PAD_SIZE_Q,
                HEAD_NUM_PER_BLOCK, HIDDEN_DIM>(
                q_ld, MAX_CTX, thread_x, warp_id, start_m, 0,
                cos_sin_table, &q_sm[0]);
        }

        float max_value[VEC_DIM_M][HEAD_NUM_PER_BLOCK];
        float sum_value[VEC_DIM_M][HEAD_NUM_PER_BLOCK];
#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
#pragma unroll
            for (int head = 0; head < HEAD_NUM_PER_BLOCK; head++) {
                max_value[mm][head] = -1e20f;
                sum_value[mm][head] = 0.0f;
            }
        }
        //

        __syncthreads();


        const int n_stage = (N + BLOCK_DIM_N - 1) / BLOCK_DIM_N;


        for (int n = 0; n < n_stage; n++) {
            int n_start = n * BLOCK_DIM_N;
            if (n_start > (start_m + BLOCK_DIM_M)) {
                continue;
            }
            float c_reg[VEC_DIM_M * VEC_DIM_N * HEAD_NUM_PER_BLOCK];
            for (int index = 0; index < HEAD_NUM_PER_BLOCK; index++) {
                for (int mm = 0; mm < VEC_DIM_M; mm++) {
                    for (int nn = 0; nn < VEC_DIM_N; nn++) {
                        c_reg[mm * VEC_DIM_N * HEAD_NUM_PER_BLOCK + nn + index * VEC_DIM_N] = 0.0f;
                    }
                }
            }

            int d = 0;
            const int k_size = min(BLOCK_DIM_K_Q, D * HEAD_NUM_PER_BLOCK - d);

            cp_async_wait_all();

            compute_tile_attention_gemm<T_KV, VEC_DIM_M, VEC_DIM_N_CM, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K_Q,
                HIDDEN_DIM,
                HEAD_NUM_PER_BLOCK, B, MBit, S, PAD_SIZE_Q, PAD_SIZE_KV>(
                N, k_size, thread_x, warp_id, start_m, n_start, q_sm, &sm[(flip_flag) * shared_mem_block_size],
                mask, c_reg);

            //

            const int next_n = (n + 1) * BLOCK_DIM_N;
            if (next_n < N) {
                //
                load_tile_kv<T_KV, VEC_DIM_N, VEC_DIM_K_KV, BLOCK_DIM_N, BLOCK_DIM_K_KV, ELEMENTS_PER_LOAD_KV, B, MBit,
                    S, PAD_SIZE_KV>(
                    kv_stride, N, thread_x, warp_id, next_n, 0,
                    k_ptr, &sm[(1 - flip_flag) * shared_mem_block_size]);
                if constexpr (rope_flag == 1) {
                    //
                    rope_k<VEC_DIM_N, VEC_DIM_K_KV, BLOCK_DIM_N, BLOCK_DIM_K_KV, ELEMENTS_PER_LOAD_KV, B, MBit, S,
                        PAD_SIZE_KV>(
                        q_ld, MAX_CTX, thread_x, warp_id, next_n, 0,
                        cos_sin_table, &sm[(1 - flip_flag) * shared_mem_block_size]);
                }
            }
            __syncthreads();
            //
            load_tile_kv<T_KV, VEC_DIM_N, VEC_DIM_K_KV, BLOCK_DIM_N, BLOCK_DIM_K_KV, ELEMENTS_PER_LOAD_KV, B, MBit, S,
                PAD_SIZE_KV>(
                kv_stride, N, thread_x, warp_id, n_start, d,
                v_ptr, &sm[(flip_flag) * shared_mem_block_size]);
            __syncthreads();

            compute_softmax_pv<VEC_DIM_M, VEC_DIM_N_CM, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K_Q, HIDDEN_DIM,
                HEAD_NUM_PER_BLOCK,
                B,
                MBit
                , S, PAD_SIZE_Q, PAD_SIZE_KV>(
                N, q_stride, start_m, n_start, d, thread_x, warp_id, &max_value[0][0],
                &sum_value[0][0], &sm[(flip_flag) * shared_mem_block_size], scale, &c_reg[0], &out_sm[0]);
            flip_flag ^= 1;
        }

        __syncthreads();
        store<VEC_DIM_M, VEC_DIM_K_Q, BLOCK_DIM_M, BLOCK_DIM_K_Q, ELEMENTS_PER_LOAD_KV, PAD_SIZE_Q>(
            M, q_stride, start_m, warp_id, thread_x, out_sm, output);
    }


    template<typename T, const int ROPE_FLAG>
    void flash_attention_128(const int max_ctx, const int batch, const int M, const int N, const int D,
                             const int q_ld, const int k_ld, const int v_ld,
                             const float scale,
                             const int num_q_heads, const int num_kv_heads,
                             const half *q_gpu,
                             const half *k_gpu,
                             const half *v_gpu,
                             half *mask,
                             float *cos_sin_table,
                             T *out_put,
                             std::shared_ptr<core::device::DeviceStream> &stream) {
        if (std::is_same_v<T, float>) {
            if (num_q_heads == 32 && num_kv_heads == 8 && D == 128) {
                constexpr int B = 4;
                constexpr int MBit = 2;
                constexpr int S = 3;
                constexpr int HEAD_NUM_PER_BLOCK = 4;
                constexpr int HIDDEN_DIM = 128;
                constexpr int BYTES_PER_LOAD = 16; // 128-bit
                constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD / (sizeof(half));
                constexpr int WARP_SIZE = 32;
                constexpr int THREAD_BLOCK_SIZE = 256;
                constexpr int BLOCK_DIM_N = 32;
                constexpr int BLOCK_DIM_M = 8;

                dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, num_q_heads, batch);
                dim3 block(THREAD_BLOCK_SIZE);

                flash_attention_fp16_8x32<half, half, 32, 8, HIDDEN_DIM, BLOCK_DIM_M, BLOCK_DIM_N,
                            ELEMENTS_PER_LOAD, ROPE_FLAG, B, MBit, S, HEAD_NUM_PER_BLOCK, WARP_SIZE, THREAD_BLOCK_SIZE>
                        <<<grid,
                        block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>>(
                            M, N, D, max_ctx, D, D, D,
                            scale,
                            q_gpu, k_gpu, v_gpu, mask, cos_sin_table, out_put);
            }
        } else if (std::is_same_v<T, float>) {
            //todo
        }
    }
    template<typename T>
    class FlashAttn<T, core::device::GPUTag> : public base::OPCreatorBase<FlashAttn<T, core::device::GPUTag>, T, core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT;
        }
    };
    //
    template<typename T>
    void tff::kernel::FlashAttn<T, core::device::GPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto max_ctx = kernel::base::get_param_value<const int>(0, para_ptr);
        auto q_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            FlashAttnBuilder::Params::Q, para_ptr);
        auto k_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            FlashAttnBuilder::Params::K, para_ptr);
        auto v_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            FlashAttnBuilder::Params::V, para_ptr);
        auto rope_table = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            PreRopeTableBuilder::Params::RopeTable, para_ptr);
        auto mask_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            FlashAttnBuilder::Params::Mask, para_ptr);
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            FlashAttnBuilder::Params::Out, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        kernel::builder::OpParamBuilderBase<FlashAttnBuilder>::CommonParams::Stream, para_ptr);

        if (q_tensor== nullptr || k_tensor == nullptr || v_tensor == nullptr || output_tensors == nullptr
            || rope_table == nullptr || mask_tensor == nullptr ) {
            return;
        }
        if (q_tensor->get_buffer() == nullptr || k_tensor->get_buffer() == nullptr || v_tensor->get_buffer() == nullptr ||
            rope_table->get_buffer() == nullptr || mask_tensor->get_buffer() == nullptr) {
            return;
        }
        auto &output = output_tensors;
        const int num_q_heads = q_tensor->get_shape()[2];
        const int num_kv_heads = k_tensor->get_shape()[2];
        const int M = q_tensor->get_shape()[1];
        const int N = k_tensor->get_shape()[1];
        const int D = q_tensor->get_shape()[0];
        const int B = q_tensor->get_shape()[3];
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));
        switch (D) {
            case 128: {
                auto *cos_sin_table = static_cast<float *>(rope_table->get_buffer()->ptr());
                if (rope_table != nullptr) {

                    flash_attention_128<T, 1>(max_ctx, B, M, N, D, D, D, D, scale, num_q_heads, num_kv_heads,
                                              static_cast<half *>(q_tensor->get_buffer()->ptr()),
                                              static_cast<half *>(k_tensor->get_buffer()->ptr()),
                                              static_cast<half *>(v_tensor->get_buffer()->ptr()),
                                              static_cast<half *>(mask_tensor->get_buffer()->ptr()),
                                              cos_sin_table,
                                              static_cast<T *>(output->get_buffer()->ptr()), stream);
                }
                break;
            }
            case 64:
                break;
            case 256:
                break;
            default:
                break;
        }
    }


    template class tff::kernel::FlashAttn<float, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(FlashAttn, float, core::device::GPUTag);
}
