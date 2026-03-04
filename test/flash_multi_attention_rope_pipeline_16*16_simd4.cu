// //
// // Created by nkk on 2026/1/14.
// //
// #include <vector>
// #include <random>
// #include "cublas_v2.h"
// #include "mma.h"
// #include <cstdint>
// #include <cstring>
// #include <algorithm>
// #include <math.h>
// #include <cmath>
// #include <cooperative_groups.h>
// #include <cuda_pipeline_primitives.h>
// #include <cuda/pipeline>
// #include <cuda/barrier>
// #define _PTX
// //#define _SWIZZLE
// #define _ASYNC
// using T_KV = half;
// using T_Q = half;
// #if 0
// constexpr int BYTES_PER_LOAD = 16; // 128-bit
// constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD / sizeof(T);
// constexpr int VEC_DIM_LOAD = 2 * ELEMENTS_PER_LOAD;
// constexpr int WARP_SIZE = 32;
// constexpr int THREAD_BLOCK_SIZE = 256;
// constexpr int BLOCK_DIM_K = 16;
// constexpr int VEC_DIM_N = 8;
// constexpr int VEC_DIM_K = 1;
// constexpr int VEC_DIM_M = 8;
// constexpr int BLOCK_DIM_M = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_M;
// constexpr int BLOCK_DIM_N = THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_N;
// constexpr int PAD_SIZE = 16; //BLOCK_DIM_K;
// #else
// constexpr int B = 4;
// constexpr int MBit = 2;
// constexpr int S = 3;
// constexpr int HEAD_NUM_PER_BLOCK = 1;
// constexpr int HIDDEN_DIM = 128;
// constexpr int BYTES_PER_LOAD = 16; // 128-bit
// constexpr int THREAD_PER_WARP_DIRECTION = 16;
// constexpr int ELEMENTS_PER_LOAD = BYTES_PER_LOAD / (sizeof(half));
// constexpr int WARP_SIZE = 32;
// constexpr int THREAD_BLOCK_SIZE = 256;
// constexpr int WARP_PER_BLOCK = THREAD_BLOCK_SIZE / THREAD_PER_WARP_DIRECTION;
// constexpr int BLOCK_DIM_K = HIDDEN_DIM * HEAD_NUM_PER_BLOCK;
// constexpr int BLOCK_DIM_M = 32; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_M;
// constexpr int BLOCK_DIM_N = 32; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_N;
// constexpr int VEC_DIM_N = BLOCK_DIM_N / THREAD_PER_WARP_DIRECTION;
// constexpr int VEC_DIM_K = BLOCK_DIM_K / THREAD_PER_WARP_DIRECTION;
// constexpr int VEC_DIM_M = BLOCK_DIM_M / WARP_PER_BLOCK;
//
// constexpr int PAD_SIZE = 4; //BLOCK_DIM_K;
// #endif
// template<int B, int M, int S = B>
// __device__ int swizzle(const int &offset) {
//     const int bit_msk = (1 << B) - 1;
//     const int yyy_msk = bit_msk << (M + max(0, S));
//     const int zzz_msk = bit_msk << (M - min(0, S));
//     const int msk_sft = S;
//     if constexpr (S >= 0) {
//         return offset ^ ((offset & yyy_msk) >> S);
//     } else {
//         return offset ^ ((offset & yyy_msk) << -S);
//     }
// }
//
// static __device__ __forceinline__ void cp_async_wait_all() {
//     asm volatile("cp.async.wait_all;");
// }
//
// static __device__ __forceinline__ void cp_async_commit() {
//     asm volatile("cp.async.commit_group;");
// }
//
// template<int preload>
// static __device__ __forceinline__ void cp_async_cg_16(const unsigned int dst, const void *src) {
//     if (preload == 256) {
//         asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], 16;"
//             : : "r"(dst), "l"(src));
//     } else if (preload == 128) {
//         asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
//             : : "r"(dst), "l"(src));
//     } else if (preload == 64) {
//         asm volatile("cp.async.cg.shared.global.L2::64B [%0], [%1], 16;"
//             : : "r"(dst), "l"(src));
//     } else {
//         asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
//             : : "r"(dst), "l"(src));
//     }
// }
//
// template<typename T1, typename T2>
// __device__ __forceinline__ void load_vec(const T1 *addr, T2 *out, const int count);
//
// template<>
// __device__ __forceinline__ void load_vec<half, half2>(const half *addr, half2 *out, const int count) {
//     if (count == 4) {
//         const auto *h = reinterpret_cast<const half2 *>(addr);
//         out[0] = h[0];
//         out[1] = h[1];
//         out[2] = h[2];
//         out[3] = h[3];
//     } else {
//         const auto *h = reinterpret_cast<const half2 *>(addr);
//         out[0] = h[0];
//     }
// }
//
// template<>
// __device__ __forceinline__ void load_vec<float, float2>(const float *addr, float2 *out, const int count) {
//     if (count == 4) {
//         const float2 *h = reinterpret_cast<const float2 *>(addr);
//         out[0] = h[0];
//         out[1] = h[1];
//         out[2] = h[2];
//         out[3] = h[3];
//     } else {
//         const float2 *h = reinterpret_cast<const float2 *>(addr);
//         out[0] = h[0];
//     }
// }
//
// template<typename T1, typename T2>
// __device__ __forceinline__ void load_vec_async(const T1 *addr, T2 *out);
//
// template<>
// __device__ __forceinline__ void load_vec_async<half, half2>(const half *addr, half2 *out) {
//     if (reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
// #ifndef _PTX
//         const half2 *h = reinterpret_cast<const half2 *>(addr);
//         out[0] = h[0];
// #else
//         const unsigned int dst_addr = __cvta_generic_to_shared(out);
//         //printf("dst_addr = %u\n", dst_addr);
//         cp_async_cg_16<64>(dst_addr, addr);
//         cp_async_commit();
// #endif
//     }
// }
//
// template<>
// __device__ __forceinline__ void load_vec_async<float, float2>(const float *addr, float2 *out) {
//     if (reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
// #ifndef _PTX
//         const float2 *h = reinterpret_cast<const float2 *>(addr);
//         out[0] = h[0];
// #else
//         const unsigned int dst_addr = __cvta_generic_to_shared(out);
//         cp_async_cg_16<64>(dst_addr, addr);
//         cp_async_commit();
// #endif
//     }
// }
//
// template<>
// __device__ __forceinline__ void load_vec_async<float, half2>(const float *addr, half2 *out) {
//     if (reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
// #ifndef _PTX
//         const float2 *h = reinterpret_cast<const float2 *>(addr);
//         out[0] = __float22half2_rn(h[0]);
// #else
//         const unsigned int dst_addr = __cvta_generic_to_shared(out);
//         cp_async_cg_16<64>(dst_addr, addr);
//         cp_async_commit();
// #endif
//     }
// }
//
// __device__ __forceinline__ half2 complex_mul_half2(const half2 &a, const half2 &b) {
//     half2 res;
//     res.x = __hsub(__hmul(a.x, b.x), __hmul(a.y, b.y));
//     res.y = __hadd(__hmul(a.x, b.y), __hmul(a.y, b.x));
//     return res;
// }
//
// template<const int ELEMENTS_PER_LOAD>
// __device__ __forceinline__ void rote(const int addr, half2 *sm, float2 *cos_sin_table) {
// #ifdef _ASYNC
//     cp_async_wait_all();
//     auto val_vec = reinterpret_cast<float4 *>(&sm[addr]);
//     float4 sm_val = val_vec[0];
//     half2 *rot_value = reinterpret_cast<half2 *>(&sm_val);
// #pragma unroll
//     for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
//         half2 val = rot_value[i];
//         rot_value[i] = complex_mul_half2(val, __float22half2_rn(cos_sin_table[i]));
//     }
// #else
//     auto val_vec = reinterpret_cast<float4 *>(&sm[addr]);
//     half2 rot_value[ELEMENTS_PER_LOAD / 2];
// #pragma unroll
//     for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
//         //half2 val = rot_value[i];
//         rot_value[i] = complex_mul_half2(val[i], __float22half2_rn(cos_sin_table_val[i]));
//     }
// #endif
//
//     float4 k_rot = *reinterpret_cast<float4 *>(&rot_value[0]);
//     val_vec[0] = k_rot;
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K,
//     const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const int
//     S>
// __device__ __forceinline__ void rope(const int ld, const int dim,
//                                      const int thread_x, const int warp_id,
//                                      const int start_m,
//                                      const int k,
//                                      const float *__restrict__ cos_sin_table,
//                                      half2 *q_sm) {
//     //rot;
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
//         for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
//             const int dim1 = k + thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
//                              ELEMENTS_PER_LOAD;
//
//             int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
//             int sm_col = thread_x * ELEMENTS_PER_LOAD / 2 + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD / 2;
//             int offset = sm_row * (BLOCK_DIM_K / 2 + PAD_SIZE) + sm_col;
// #ifdef _SWIZZLE
//             int addr8 = swizzle<B, MBit, S>(offset);
// #else
//             int addr8 = offset; //
// #endif
//
//
//             float2 cos_sin_table_val[ELEMENTS_PER_LOAD / 2];
//             if (dim0_base < dim) {
//                 const int actual_load = min(ELEMENTS_PER_LOAD / 2, ld - dim1);
//                 if (actual_load > 0) {
//                     load_vec<float, float2>(&cos_sin_table[dim0_base * ld + dim1], &cos_sin_table_val[0],
//                                             actual_load);
//                 }
//             } else {
// #pragma unroll
//                 for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
//                     cos_sin_table_val[i].x = 0;
//                     cos_sin_table_val[i].y = 0;
//                 }
//             }
//
//             //
//             rote<ELEMENTS_PER_LOAD>(addr8, q_sm, cos_sin_table_val);
//         }
//     }
// }
//
// template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
//     const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const int
//     S>
// __device__ __forceinline__ void load_tile(const int ld, const int dim,
//                                           const int thread_x, const int warp_id,
//                                           const int start_m,
//                                           const int k,
//                                           const T *__restrict__ global_mem,
//                                           half2 *sm) {
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
//         for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
//             const int dim1 = k + thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) *
//                              ELEMENTS_PER_LOAD;
//
//             int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
//             int sm_col = thread_x * ELEMENTS_PER_LOAD / 2 + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD / 2;
//             int offset = sm_row * (BLOCK_DIM_K / 2 + PAD_SIZE) + sm_col;
// #ifdef _SWIZZLE
//             int addr8 = swizzle<B, MBit, S>(offset);
// #else
//             int addr8 = offset; //
// #endif
// #ifndef _ASYNC
//             half2 val[ELEMENTS_PER_LOAD / 2];
// #endif
//             if (dim0_base < dim) {
//                 const int actual_load = min(ELEMENTS_PER_LOAD, ld - dim1);
//                 if (actual_load > 0) {
// #ifdef _ASYNC
//                     load_vec_async<T, half2>(&global_mem[dim0_base * ld + dim1], &sm[addr8]);
// #else
//                     load_vec<T, half2>(&global_mem[dim0_base * ld + dim1], &val[0], ELEMENTS_PER_LOAD / 2);
// #endif
//                 }
//             } else {
// #pragma unroll
//                 for (int i = 0; i < ELEMENTS_PER_LOAD / 2; ++i) {
//                     int addr = swizzle<B, MBit, S>(offset + i);
//                     sm[addr].x = 0;
//                     sm[addr].y = 0;
//                 }
//             }
//         }
//     }
// }
//
// template<typename T_KV, const int VEC_DIM_M, const int VEC_DIM_N,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int
//     ELEMENTS_PER_LOAD, const int B, const int MBit, const int S>
// __device__ __forceinline__ void compute_tile_attention_gemm(const int N, const int k_size,
//                                                             const int thread_x,
//                                                             const int warp_id,
//                                                             const int start_m, const int start_n,
//                                                             half2 *q_sm,
//                                                             half2 *b_sm,
//                                                             const T_KV *mask,
//                                                             float *c_reg) {
//     //#pragma unroll
//     for (int kk = 0; kk < k_size / 2; kk++) {
//         const int stride = (BLOCK_DIM_K / 2 + PAD_SIZE);
// #pragma unroll
//         for (int mm = 0; mm < VEC_DIM_M; mm++) {
//             int m_row_base = (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M);
//             int col_base = kk;
//             int q_index = m_row_base * stride + col_base;
// #ifdef _SWIZZLE
//             int q_addr = swizzle<B, MBit, S>(q_index);
// #else
//             int q_addr = q_index;
// #endif
//             float2 q_rot = __half22float2(q_sm[q_addr]);
//
// #pragma unroll
//             for (int nn = 0; nn < VEC_DIM_N; nn++) {
//                 int n_row_base = (thread_x + nn * BLOCK_DIM_N / (VEC_DIM_N));
//                 int k_index = col_base + n_row_base * stride;
//                 const float mask_value = __ldg(&mask[(start_m + m_row_base) * N + start_n + n_row_base]);
// #ifdef _SWIZZLE
//                 int k_addr = swizzle<B, MBit, S>(k_index);
// #else
//                 int k_addr = k_index;
// #endif
//                 // #ifdef _DEBUG
//                 //                 if (nn == 0 && mm == 0 && kk == 0 && start_m == 0 && start_n == 0 && warp_id == 0) {
//                 //                     printf("warp_id: %d, thread_x: %d, kk: %d,"
//                 //                            "row: %d, col: %d, swizzle addr: %d\n", warp_id, thread_x, kk, n_row_base, col_base, k_addr);
//                 //                 }
//                 // #endif
//                 float2 k_rot = __half22float2(b_sm[k_addr]);
//
//                 c_reg[mm * VEC_DIM_N + nn] += q_rot.x * k_rot.x + q_rot.y * k_rot.y + mask_value;
//
//
//                 // if (mm == 0  && warp_id == 1 && blockIdx.y == 0&& start_m == 0 && start_n == 0) {
//                 //     printf("m: %d, n: %d, k: %d, "
//                 //            "q_rot[0].x:%lf, q_rot[0].y:%lf "
//                 //            "k_rot[0].x:%lf, k_rot[0].y:%lf "
//                 //            "c_reg[mm * VEC_DIM_N + nn]:%lf "
//                 //            "mask_value: %lf \n",
//                 //         start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M, start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N,kk,
//                 //         q_rot.x, q_rot.y,
//                 //         k_rot.x, k_rot.y,
//                 //         c_reg[mm * VEC_DIM_N + nn],
//                 //         mask_value);
//                 // }
//                 // if (warp_id == 0 && start_m == 0 && start_n == 0&&  mm ==0 && blockIdx.y == 0) {
//                 //     printf("000 thread_x: %d, warp_id: %d,"
//                 //            " m: %d, n: %d, "
//                 //            "c_reg[mm * VEC_DIM_N + nn]:%lf,"
//                 //            "mask_value: %lf \n",
//                 //            thread_x, warp_id,
//                 //            start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M,
//                 //            start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N,
//                 //            c_reg[mm * VEC_DIM_N + nn],
//                 //            mask_value);
//                 // }
//             }
//         }
//     }
// }
//
// template<const int VEC_DIM_M, const int VEC_DIM_N, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int
//     BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const int S>
// __device__ __forceinline__ void compute_softmax_pv(const int N, const int v_ld, const int start_m, const int start_n,
//                                                    const int start_d,
//                                                    const int thread_x, const int warp_id,
//                                                    float *max_value, float *sum_value, half2 *sm,
//                                                    const float scale,
//                                                    float *c_reg,
//                                                    float *output) {
//     float2 *output_ptr = reinterpret_cast<float2 *>(output);
//     const int base_n = start_n + thread_x;
//     const int valid_n = min(VEC_DIM_N, (N - base_n + 31) / 32);
//     const int stride = (BLOCK_DIM_K / 2 + PAD_SIZE);
// #pragma unroll
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const float old_max = max_value[mm];
//         const float old_sum = sum_value[mm];
//         float new_max_value = {-1e20f};
//         float new_sum_value = {0.0f};
//         float current_max_value = {-1e20f};
//         float current_sum_value = {0.0f};
//
//         for (int nn = 0; nn < valid_n; nn++) {
//             // if (start_m == 0 && warp_id == 1 && thread_x == 0 && nn == 0 && mm == 1) {
//             //     printf("111 warp_id: %d, thread_x: %d, m: %d, "
//             //            "c_reg: %lf, "
//             //            " new_max_value: %lf,new_sum_value: %lf "
//             //            "current_max_value: %lf \n",
//             //                warp_id, thread_x,
//             //                (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M),
//             //                c_reg[mm * VEC_DIM_N + nn],
//             //                new_max_value,
//             //                new_sum_value);
//             // }
//             c_reg[mm * VEC_DIM_N + nn] *= scale;
//             current_max_value = fmaxf(current_max_value, c_reg[mm * VEC_DIM_N + nn]);
//         }
// #pragma unroll
//         for (int offset = 16; offset > 0; offset /= 2) {
//             current_max_value = fmaxf(current_max_value, __shfl_xor_sync(0xffffffff, current_max_value, offset, 32));
//         }
//
//         //
//         for (int nn = 0; nn < valid_n; nn++) {
//             c_reg[mm * VEC_DIM_N + nn] = expf(c_reg[mm * VEC_DIM_N + nn] - current_max_value);
//             current_sum_value += c_reg[mm * VEC_DIM_N + nn];
//         }
//
// #pragma unroll
//         for (int offset = 16; offset > 0; offset /= 2) {
//             current_sum_value += __shfl_xor_sync(0xffffffff, current_sum_value, offset, 32);
//         }
//
//         new_max_value = fmaxf(current_max_value, old_max);
//         new_sum_value = current_sum_value * expf(current_max_value - new_max_value) + old_sum * expf(
//                             old_max - new_max_value);
//
//         for (int kk = start_d; kk < BLOCK_DIM_K / 2; kk++) {
//             float2 acc;
//             acc.x = 0;
//             acc.y = 0;
//
//             for (int nn = 0; nn < valid_n; nn++) {
//                 const int n_row_base = (thread_x + nn * BLOCK_DIM_N / (VEC_DIM_N));
//                 int index = kk + n_row_base * stride;
// #ifdef _SWIZZLE
//                 int addr = swizzle<B, MBit, S>(index);
// #else
//                 int addr = index;
// #endif
//                 float2 v_reg_0 = __half22float2(sm[addr]);
//
//                 acc.x += v_reg_0.x * c_reg[mm * VEC_DIM_N + nn];
//                 acc.y += v_reg_0.y * c_reg[mm * VEC_DIM_N + nn];
//                 // if (start_m == 0 && kk == 0 && warp_id == 1 && mm == 0 && blockIdx.y == 0) {
//                 //     printf("222 warp_id: %d, thread_x: %d, m: %d, k: %d, n: %d, "
//                 //            "acc.x: %lf, acc.y: %lf "
//                 //            "v_reg_0.x: %lf, v_reg_0.y: %lf "
//                 //            "c_reg: %lf, "
//                 //            " new_max_value: %lf,new_sum_value: %lf "
//                 //            "current_max_value: %lf, current_sum_value: %lf, scale: %lf\n",
//                 //                warp_id, thread_x,
//                 //                (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M), kk,n_row_base,
//                 //                acc.x,acc.y,
//                 //                v_reg_0.x, v_reg_0.y,c_reg[mm * VEC_DIM_N + nn],
//                 //                new_max_value,
//                 //                new_sum_value,
//                 //                current_max_value,
//                 //                current_sum_value, scale);
//                 // }
//             }
// #pragma unroll
//             for (int offset = 16; offset > 0; offset /= 2) {
//                 acc.x += __shfl_xor_sync(0xffffffff, acc.x, offset, 32);
//                 acc.y += __shfl_xor_sync(0xffffffff, acc.y, offset, 32);
//             }
//
//             int output_index = (start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld / 2 + kk;
//
//
//             output_ptr[output_index].x = (output_ptr[output_index].x * old_sum * expf(old_max - new_max_value) +
//                                           acc.x * expf(current_max_value - new_max_value)) / new_sum_value;
//
//             output_ptr[output_index].y = (output_ptr[output_index].y * old_sum * expf(old_max - new_max_value) +
//                                           acc.y * expf(current_max_value - new_max_value)) / new_sum_value;
//
//             // if (start_m == 0 && warp_id == 1 && thread_x == 0 && mm == 0&& blockIdx.y == 0) {
//             //     printf("333 warp_id: %d, thread_x: %d, "
//             //            "m: %d, k: %d, "
//             //            "acc.x: %lf, acc.y: %lf "
//             //            "new_max_value: %lf,new_sum_value: %lf "
//             //            "old_sum: %lf, old_max: %lf,"
//             //            "current_max_value: %lf, current_sum_value: %lf, "
//             //            "result.x: %lf, result.y: %lf \n",
//             //                warp_id, thread_x,
//             //                (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M), kk,
//             //                acc.x,acc.y,
//             //                new_max_value, new_sum_value,
//             //                old_sum, old_max,
//             //                current_max_value,
//             //                current_sum_value,
//             //                output_ptr[output_index].x,
//             //                output_ptr[output_index].y);
//             // }
//         }
//         max_value[mm] = new_max_value;
//         sum_value[mm] = new_sum_value;
//     }
// }
//
// template<const int VEC_DIM_M, const int VEC_DIM_N, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int
//     BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int B, const int MBit, const int S>
// __device__ __forceinline__ void compute_softmax_pv_sm(const int N, const int v_ld, const int start_m, const int start_n,
//                                                       const int start_d,
//                                                       const int thread_x, const int warp_id,
//                                                       float *max_value, float *sum_value, half2 *sm,
//                                                       const float scale,
//                                                       float *c_reg,
//                                                       half2 *output_sm) {
//     half2 *output_ptr = reinterpret_cast<half2 *>(output_sm);
//     const int base_n = start_n + thread_x;
//     const int valid_n = min(VEC_DIM_N, (N - base_n + 31) / 32);
//     const int stride = (BLOCK_DIM_K / 2 + PAD_SIZE);
// #pragma unroll
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const float old_max = max_value[mm];
//         const float old_sum = sum_value[mm];
//         float new_max_value = {-1e20f};
//         float new_sum_value = {0.0f};
//         float current_max_value = {-1e20f};
//         float current_sum_value = {0.0f};
//
//         for (int nn = 0; nn < valid_n; nn++) {
//             c_reg[mm * VEC_DIM_N + nn] *= scale;
//             current_max_value = fmaxf(current_max_value, c_reg[mm * VEC_DIM_N + nn]);
//         }
// #pragma unroll
//         for (int offset = 16; offset > 0; offset /= 2) {
//             current_max_value = fmaxf(current_max_value, __shfl_xor_sync(0xffffffff, current_max_value, offset, 32));
//         }
//
//         //
//         for (int nn = 0; nn < valid_n; nn++) {
//             c_reg[mm * VEC_DIM_N + nn] = expf(c_reg[mm * VEC_DIM_N + nn] - current_max_value);
//             current_sum_value += c_reg[mm * VEC_DIM_N + nn];
//         }
//
// #pragma unroll
//         for (int offset = 16; offset > 0; offset /= 2) {
//             current_sum_value += __shfl_xor_sync(0xffffffff, current_sum_value, offset, 32);
//         }
//
//         new_max_value = fmaxf(current_max_value, old_max);
//         new_sum_value = current_sum_value * expf(current_max_value - new_max_value) + old_sum * expf(
//                             old_max - new_max_value);
//
//         for (int kk = start_d; kk < BLOCK_DIM_K / 2; kk++) {
//             float2 acc;
//             acc.x = 0;
//             acc.y = 0;
//
//             for (int nn = 0; nn < valid_n; nn++) {
//                 const int n_row_base = (thread_x + nn * BLOCK_DIM_N / (VEC_DIM_N));
//                 int index = kk + n_row_base * stride;
// #ifdef _SWIZZLE
//                 int addr = swizzle<B, MBit, S>(index);
// #else
//                 int addr = index;
// #endif
//                 float2 v_reg_0 = __half22float2(sm[addr]);
//
//                 acc.x += v_reg_0.x * c_reg[mm * VEC_DIM_N + nn];
//                 acc.y += v_reg_0.y * c_reg[mm * VEC_DIM_N + nn];
//             }
// #pragma unroll
//             for (int offset = 16; offset > 0; offset /= 2) {
//                 acc.x += __shfl_xor_sync(0xffffffff, acc.x, offset, 32);
//                 acc.y += __shfl_xor_sync(0xffffffff, acc.y, offset, 32);
//             }
//
//             int output_index = (warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * stride + kk;
//
//
//             output_ptr[output_index].x = __float2half(
//                 __half2float(output_ptr[output_index].x) * old_sum * expf(old_max - new_max_value) +
//                 acc.x * expf(current_max_value - new_max_value) / new_sum_value);
//
//             output_ptr[output_index].y = __float2half(
//                 __half2float(output_ptr[output_index].y) * old_sum * expf(old_max - new_max_value) +
//                 acc.y * expf(current_max_value - new_max_value) / new_sum_value);
//         }
//         max_value[mm] = new_max_value;
//         sum_value[mm] = new_sum_value;
//     }
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K>
// __device__ __forceinline__ void init(
//     const int thread_x, const int warp_id,
//     half2 *sm, const half value) {
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         //VEC_DIM_LD = 4;BLOCK_DIM_LD = 32;
//         int sm_row = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
//         for (int kk = 0; kk < VEC_DIM_K; ++kk) {
//             //VEC_DIM_K = 2;BLOCK_DIM_K = 64
//             int sm_col = thread_x + kk * (BLOCK_DIM_K / VEC_DIM_K);
//             int offset = sm_row * (BLOCK_DIM_K + PAD_SIZE) + sm_col;
//             sm[offset].x = value;
//             sm[offset].y = value;
//         }
//     }
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int PAD_SIZE>
// __device__ __forceinline__ void store(const int M,
//                                       const int ld,
//                                       const int start_m,
//                                       const int warp_id, const int thread_x,
//                                       half2 *output_sm,
//                                       float *out_put) {
//     float2 *output_ptr = reinterpret_cast<float2 *>(out_put);
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int row_base = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
// #pragma unroll
//         for (int kk = 0; kk < VEC_DIM_K; ++kk) {
//             int col_base = thread_x + kk * (BLOCK_DIM_K / VEC_DIM_K);
//             float2 out = __half22float2(output_sm[row_base * (BLOCK_DIM_K + PAD_SIZE) + col_base]);
//             if ((start_m + row_base) < M) {
//                 output_ptr[(start_m + row_base) * ld / 2 + col_base] = out;
//             }
//         }
//     }
// }
//
// template<typename T_Q, typename T_KV, const int VEC_DIM_M, const int VEC_DIM_N,
//     const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int ELEMENTS_PER_LOAD,
//     const int rope_flag, const int B, const int MBit, const int S, const int HEAD_NUM_PER_BLOCK,
//     const int WARP_SIZE, const int THREAD_BLOCK>
// __global__ void flash_attention_double_buffer_k_out_sm(const int M, const int N, const int D,
//                                                        const int MAX_CTX,
//                                                        const int q_ld, const int k_ld, const int v_ld,
//                                                        const float scale,
//                                                        const int num_q_heads, const int num_kv_heads,
//                                                        const T_Q *__restrict__ q_global,
//                                                        const T_KV *__restrict__ k_global,
//                                                        const T_KV *__restrict__ v_global,
//                                                        const T_KV *mask,
//                                                        float *__restrict__ cos_sin_table,
//                                                        float *out_put) {
//     const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//     const int ld_thread_block_n = BLOCK_DIM_K / VEC_DIM_K;
//     const int ld_thread_x = thread_id % ld_thread_block_n;
//     const int ld_thread_y = thread_id / ld_thread_block_n;
//     const int cm_thread_x = thread_id % WARP_SIZE;
//     const int cm_thread_y = thread_id / WARP_SIZE;
//     const int K_STRIDE = BLOCK_DIM_K / 2;
//     const int VEC_DIM_M_CM = BLOCK_DIM_M / (THREAD_BLOCK / WARP_SIZE);
//     const int VEC_DIM_N_CM = BLOCK_DIM_N / WARP_SIZE;
//     const int VEC_DIM_K_CM = K_STRIDE / WARP_SIZE;
//
//     const int block_x = blockIdx.x;
//     const int start_m = block_x * BLOCK_DIM_M;
//
//     const int head_index = blockIdx.y * HEAD_NUM_PER_BLOCK;
//     const int kv_group_per_q = num_q_heads / num_kv_heads;
//     const int kv_group_start_index = head_index / kv_group_per_q;
//
//     const int kv_stride = num_kv_heads * k_ld;
//     const int q_stride = num_q_heads * q_ld;
//
//     const T_Q *q_ptr = q_global + (blockIdx.z * num_q_heads) * M * D + head_index * D;
//     const T_KV *k_ptr = k_global + (blockIdx.z * num_kv_heads) * N * D + kv_group_start_index * D;
//     const T_KV *v_ptr = v_global + (blockIdx.z * num_kv_heads) * N * D + kv_group_start_index * D;
//     float *output = out_put + (blockIdx.z * num_q_heads) * M * D + head_index * D;
//
//     constexpr int shared_mem_block_size = (BLOCK_DIM_K / 2 + PAD_SIZE) * (BLOCK_DIM_N);
//     __shared__ half2 sm[shared_mem_block_size * 2]; //2 * kv;(k, v共享一块shared mem)
//
//     constexpr int q_sm_size = (BLOCK_DIM_K / 2 + PAD_SIZE) * (BLOCK_DIM_M);
//     __shared__ half2 q_sm[q_sm_size];
//
//     constexpr int out_sm_size = (BLOCK_DIM_K / 2 + PAD_SIZE) * (BLOCK_DIM_M);
//     __shared__ half2 out_sm[out_sm_size];
//     init<VEC_DIM_M_CM, VEC_DIM_K_CM, BLOCK_DIM_M, K_STRIDE>(cm_thread_x, cm_thread_y, out_sm, 0);
//
//     int flip_flag = 0;
//     load_tile<T_KV, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//         kv_stride, N, ld_thread_x, ld_thread_y, 0, 0,
//         k_ptr, &sm[(flip_flag) * shared_mem_block_size]);
//
//     load_tile<T_Q, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//         q_stride, M, ld_thread_x, ld_thread_y, start_m, 0,
//         q_ptr, &q_sm[0]);
//
//
//     if constexpr (rope_flag == 1) {
//         rope<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//             q_ld, MAX_CTX, ld_thread_x, ld_thread_y, 0, 0,
//             cos_sin_table, &sm[(flip_flag) * shared_mem_block_size]);
//
//         rope<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//             q_ld, MAX_CTX, ld_thread_x, ld_thread_y, 0, 0,
//             cos_sin_table, &q_sm[0]);
//     }
//
//     float max_value[VEC_DIM_M_CM];
//     float sum_value[VEC_DIM_M_CM];
// #pragma unroll
//     for (int mm = 0; mm < VEC_DIM_M_CM; mm++) {
//         max_value[mm] = -1e20f;
//         sum_value[mm] = 0.0f;
//     }
//
//     __syncthreads();
//     //__syncwarp();
//
//     const int n_stage = (N + BLOCK_DIM_N - 1) / BLOCK_DIM_N;
//
//
//     for (int n = 0; n < n_stage; n++) {
//         int n_start = n * BLOCK_DIM_N;
//         if (n_start > (start_m + BLOCK_DIM_M)) {
//             continue;
//         }
//         float c_reg[VEC_DIM_M_CM * VEC_DIM_N_CM] = {0};
//
//         int d = 0;
//         const int k_size = min(BLOCK_DIM_K, D - d);
//         cp_async_wait_all();
//         compute_tile_attention_gemm<T_KV, VEC_DIM_M_CM, VEC_DIM_N_CM, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//             ELEMENTS_PER_LOAD, B, MBit, S>(
//             N, k_size, cm_thread_x, cm_thread_y, start_m, n_start, q_sm, &sm[(flip_flag) * shared_mem_block_size],
//             mask, c_reg);
//
//         const int next_n = (n + 1) * BLOCK_DIM_N;
//         if (next_n < N) {
//             load_tile<T_KV, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//                 kv_stride, N, ld_thread_x, ld_thread_y, next_n, 0,
//                 k_ptr, &sm[(1 - flip_flag) * shared_mem_block_size]);
//             if constexpr (rope_flag == 1) {
//                 //
//                 rope<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//                     q_ld, MAX_CTX, ld_thread_x, ld_thread_y, 0, 0,
//                     cos_sin_table, &sm[(1 - flip_flag) * shared_mem_block_size]);
//             }
//         }
//         __syncthreads();
//
//         load_tile<T_KV, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//             kv_stride, N, ld_thread_x, ld_thread_y, n_start, d,
//             v_ptr, &sm[(flip_flag) * shared_mem_block_size]);
//
//         cp_async_wait_all();
//         __syncthreads();
//         compute_softmax_pv<VEC_DIM_M_CM, VEC_DIM_N_CM, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit
//              , S>(
//              N, q_stride, start_m, n_start, d, cm_thread_x, cm_thread_y, &max_value[0],
//              &sum_value[0], &sm[(flip_flag) * shared_mem_block_size], scale, &c_reg[0], output);
//         // compute_softmax_pv_sm<VEC_DIM_M_CM, VEC_DIM_N_CM, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B,
//         //     MBit
//         //     , S>(
//         //     N, q_stride, start_m, n_start, d, cm_thread_x, cm_thread_y, &max_value[0],
//         //     &sum_value[0], &sm[(flip_flag) * shared_mem_block_size], scale, &c_reg[0], out_sm);
//         //__syncthreads();
//         flip_flag ^= 1;
//     }
//     __syncthreads();
//     //
//     // store<VEC_DIM_M_CM, VEC_DIM_K_CM, BLOCK_DIM_M, BLOCK_DIM_K / 2, PAD_SIZE>(
//     //     M, q_stride, start_m, cm_thread_y, cm_thread_x, out_sm, output);
// }
//
// template<typename T_Q, typename T_KV, const int VEC_DIM_M, const int VEC_DIM_N,
//     const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int ELEMENTS_PER_LOAD,
//     const int rope_flag, const int B, const int MBit, const int S, const int HEAD_NUM_PER_BLOCK,
//     const int WARP_SIZE, const int THREAD_BLOCK>
// __global__ void flash_attention_double_buffer_k(const int M, const int N, const int D,
//                                                 const int MAX_CTX,
//                                                 const int q_ld, const int k_ld, const int v_ld,
//                                                 const float scale,
//                                                 const int num_q_heads, const int num_kv_heads,
//                                                 const T_Q *__restrict__ q_global,
//                                                 const T_KV *__restrict__ k_global,
//                                                 const T_KV *__restrict__ v_global,
//                                                 const T_KV *mask,
//                                                 float *__restrict__ cos_sin_table,
//                                                 float *out_put) {
//     const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//     const int ld_thread_block_n = BLOCK_DIM_K / VEC_DIM_K;
//     const int ld_thread_x = thread_id % ld_thread_block_n;
//     const int ld_thread_y = thread_id / ld_thread_block_n;
//     const int cm_thread_x = thread_id % WARP_SIZE;
//     const int cm_thread_y = thread_id / WARP_SIZE;
//     const int VEC_DIM_M_CM = BLOCK_DIM_M / (THREAD_BLOCK / WARP_SIZE);
//     const int VEC_DIM_N_CM = BLOCK_DIM_N / WARP_SIZE;
//
//     const int block_x = blockIdx.x;
//     const int start_m = block_x * BLOCK_DIM_M;
//
//     const int head_index = blockIdx.y * HEAD_NUM_PER_BLOCK;
//     const int kv_group_per_q = num_q_heads / num_kv_heads;
//     const int kv_group_start_index = head_index / kv_group_per_q;
//
//     const int kv_stride = num_kv_heads * k_ld;
//     const int q_stride = num_q_heads * q_ld;
//
//     const T_Q *q_ptr = q_global + (blockIdx.z * num_q_heads) * M * D + head_index * D;
//     const T_KV *k_ptr = k_global + (blockIdx.z * num_kv_heads) * N * D + kv_group_start_index * D;
//     const T_KV *v_ptr = v_global + (blockIdx.z * num_kv_heads) * N * D + kv_group_start_index * D;
//     float *output = out_put + (blockIdx.z * num_q_heads) * M * D + head_index * D;
//
//     constexpr int shared_mem_block_size = (BLOCK_DIM_K / 2 + PAD_SIZE) * (BLOCK_DIM_N);
//     __shared__ half2 sm[shared_mem_block_size * 2]; //2 * kv;(k, v共享一块shared mem)
//
//     constexpr int q_sm_size = (BLOCK_DIM_K / 2 + PAD_SIZE) * (BLOCK_DIM_M);
//     __shared__ half2 q_sm[q_sm_size];
//
//     int flip_flag = 0;
//     load_tile<T_KV, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//         kv_stride, N, ld_thread_x, ld_thread_y, 0, 0,
//         k_ptr, &sm[(flip_flag) * shared_mem_block_size]);
//
//     load_tile<T_Q, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//         q_stride, M, ld_thread_x, ld_thread_y, start_m, 0,
//         q_ptr, &q_sm[0]);
//
//
//     if constexpr (rope_flag == 1) {
//         rope<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//             q_ld, MAX_CTX, ld_thread_x, ld_thread_y, 0, 0,
//             cos_sin_table, &sm[(flip_flag) * shared_mem_block_size]);
//
//         rope<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//             q_ld, MAX_CTX, ld_thread_x, ld_thread_y, 0, 0,
//             cos_sin_table, &q_sm[0]);
//     }
//
//     float max_value[VEC_DIM_M_CM];
//     float sum_value[VEC_DIM_M_CM];
// #pragma unroll
//     for (int mm = 0; mm < VEC_DIM_M_CM; mm++) {
//         max_value[mm] = -1e20f;
//         sum_value[mm] = 0.0f;
//     }
//
//     __syncthreads();
//     //__syncwarp();
//
//     const int n_stage = (N + BLOCK_DIM_N - 1) / BLOCK_DIM_N;
//
//
//     for (int n = 0; n < n_stage; n++) {
//         int n_start = n * BLOCK_DIM_N;
//         if (n_start > (start_m + BLOCK_DIM_M)) {
//             continue;
//         }
//         float c_reg[VEC_DIM_M_CM * VEC_DIM_N_CM] = {0};
//
//         int d = 0;
//         const int k_size = min(BLOCK_DIM_K, D - d);
//         cp_async_wait_all();
//         compute_tile_attention_gemm<T_KV, VEC_DIM_M_CM, VEC_DIM_N_CM, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//             ELEMENTS_PER_LOAD, B, MBit, S>(
//             N, k_size, cm_thread_x, cm_thread_y, start_m, n_start, q_sm, &sm[(flip_flag) * shared_mem_block_size],
//             mask, c_reg);
//
//         const int next_n = (n + 1) * BLOCK_DIM_N;
//         if (next_n < N) {
//             load_tile<T_KV, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//                 kv_stride, N, ld_thread_x, ld_thread_y, next_n, 0,
//                 k_ptr, &sm[(1 - flip_flag) * shared_mem_block_size]);
//             if constexpr (rope_flag == 1) {
//                 //
//                 rope<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//                     q_ld, MAX_CTX, ld_thread_x, ld_thread_y, 0, 0,
//                     cos_sin_table, &sm[(1 - flip_flag) * shared_mem_block_size]);
//             }
//         }
//         __syncthreads();
//
//         load_tile<T_KV, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit, S>(
//             kv_stride, N, ld_thread_x, ld_thread_y, n_start, d,
//             v_ptr, &sm[(flip_flag) * shared_mem_block_size]);
//
//         cp_async_wait_all();
//         __syncthreads();
//         compute_softmax_pv<VEC_DIM_M_CM, VEC_DIM_N_CM, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, ELEMENTS_PER_LOAD, B, MBit
//             , S>(
//             N, q_stride, start_m, n_start, d, cm_thread_x, cm_thread_y, &max_value[0],
//             &sum_value[0], &sm[(flip_flag) * shared_mem_block_size], scale, &c_reg[0], output);
//         //__syncthreads();
//         flip_flag ^= 1;
//     }
// }
//
// template<typename T>
// inline float to_float(T val) {
//     if constexpr (std::is_same_v<T, half>) {
//         return __half2float(val);
//     } else if constexpr (std::is_same_v<T, float>) {
//         return val;
//     } else {
//         static_assert(sizeof(T) == 0, "Unsupported type");
//     }
// }
//
// template<typename T, const int rope_flag>
// static void apply_rope_to_vec(
//     const T *src,
//     float *dst,
//     int head_dim,
//     int pos,
//     const float *cos_ptr,
//     const float *sin_ptr) {
//     int half = head_dim / 2;
//     for (int i = 0; i < half; ++i) {
//         float x0 = to_float(src[2 * i]);
//         float x1 = to_float(src[2 * i + 1]);
//         //printf("cpu: d: %d, x0: %lf, x1: %lf \n",i, x0, x1);
//         if (rope_flag == 1) {
//             float c = cos_ptr[pos * half + i];
//             float s = sin_ptr[pos * half + i];
//             dst[2 * i] = x0 * c - x1 * s;
//             dst[2 * i + 1] = x0 * s + x1 * c;
//             int tmp = i + 1;
//         } else {
//             dst[2 * i] = x0;
//             dst[2 * i + 1] = x1;
//         }
//     }
//     //printf("\n");
// }
//
// template<typename T_Q, typename T_KV, const int rope_flag>
// void flash_attention_cpu_single_head(
//     int m, int n, int k,
//     const int q_head, const int kv_head,
//     const int q_head_index, const int kv_head_index,
//     const T_Q *q_mat,
//     const T_KV *k_mat,
//     const T_KV *v_mat,
//     float *out_ptr,
//     const float *cos_ptr,
//     const float *sin_ptr,
//     const half *mask_ptr,
//     int block_size_m = 64,
//     int block_size_n = 64) {
//     const float scaling = 1.0f / std::sqrt(static_cast<float>(k));
//     std::vector<float> q_rot(k);
//     std::vector<float> k_rot(k);
//
//     for (int m_start = 0; m_start < m; m_start += block_size_m) {
//         int m_end = std::min(m_start + block_size_m, m);
//         int current_m = m_end - m_start;
//
//         std::vector<float> running_max(current_m, -1e20f);
//         std::vector<float> running_sum(current_m, 0.0f);
//
//         for (int n_start = 0; n_start < n; n_start += block_size_n) {
//             if (n_start > m_end) {
//                 continue;
//             }
//             int n_end = std::min(n_start + block_size_n, n);
//             int current_n = n_end - n_start;
//
//             std::vector<std::vector<float> > local_logits(current_m, std::vector<float>(current_n, 0.0f));
//
//             for (int im = 0; im < current_m; ++im) {
//                 int i = m_start + im;
//                 apply_rope_to_vec<T_Q, rope_flag>(q_mat + i * k * q_head + q_head_index * k, q_rot.data(), k, i,
//                                                   cos_ptr, sin_ptr);
//
//                 for (int j_idx = 0; j_idx < current_n; ++j_idx) {
//                     int j = n_start + j_idx;
//                     apply_rope_to_vec<T_KV, rope_flag>(k_mat + j * k * kv_head + kv_head_index * k, k_rot.data(), k, j,
//                                                        cos_ptr, sin_ptr);
//
//                     float mask_value = mask_ptr[i * n + j];
//                     float dot = 0.0f;
//                     for (int l = 0; l < k; ++l) {
//                         dot += q_rot[l] * k_rot[l];
//                     }
//                     dot += mask_value;
//                     local_logits[im][j_idx] = dot * scaling;
//                 }
//             }
//
//             std::vector<float> local_max(current_m);
//             std::vector<float> local_sum(current_m);
//             std::vector<std::vector<float> > local_exps(current_m, std::vector<float>(current_n));
//
//             for (int im = 0; im < current_m; ++im) {
//                 local_max[im] = *std::max_element(local_logits[im].begin(), local_logits[im].end());
//                 if (std::isinf(local_max[im])) {
//                     std::fill(local_exps[im].begin(), local_exps[im].end(), 0.0f);
//                     local_sum[im] = 0.0f;
//                 } else {
//                     local_sum[im] = 0.0f;
//                     for (int j_idx = 0; j_idx < current_n; ++j_idx) {
//                         local_exps[im][j_idx] = ::expf(local_logits[im][j_idx] - local_max[im]);
//                         local_sum[im] += local_exps[im][j_idx];
//                     }
//                 }
//             }
//
//             for (int im = 0; im < current_m; ++im) {
//                 int i = m_start + im;
//                 float old_max = running_max[im];
//                 float old_sum = running_sum[im];
//                 float new_max = std::max(old_max, local_max[im]);
//                 float exp_old = ::expf(old_max - new_max);
//                 float exp_local = ::expf(local_max[im] - new_max);
//                 float new_sum = running_sum[im] * exp_old + local_sum[im] * exp_local;
//
//                 for (int l = 0; l < k; ++l) {
//                     float acc = 0.0f;
//                     for (int j_idx = 0; j_idx < current_n; ++j_idx) {
//                         int j = n_start + j_idx;
//                         const float v_reg = to_float(v_mat[j * k * kv_head + kv_head_index * k + l]);
//
//                         acc += local_exps[im][j_idx] * v_reg;
//                     }
//                     float corrected_acc = (exp_old * out_ptr[i * k * q_head + q_head_index * k + l] * old_sum + acc *
//                                            exp_local) / new_sum;
//                     out_ptr[i * k * q_head + q_head_index * k + l] = corrected_acc;
//                 }
//
//                 running_max[im] = new_max;
//                 running_sum[im] = new_sum;
//             }
//         }
//         // for (int mm = 0; mm < current_m; ++mm) {
//         //     int i = m_start + mm;
//         //     if (running_sum[mm] != 0.0f) {
//         //         for (int l = 0; l < k; ++l) {
//         //             out_ptr[i * k * q_head + q_head_index * k + l] /= running_sum[mm];
//         //         }
//         //     }
//         // }
//     }
// }
//
// template<typename T_Q, typename T_KV, const int rope_flag>
// std::vector<float> flash_attention_cpu_gqa(
//     int batch,
//     int m, int n, int head_dim,
//     int num_q_heads,
//     int num_kv_heads,
//     const std::vector<T_Q> &q_mat,
//     const std::vector<T_KV> &k_mat,
//     const std::vector<T_KV> &v_mat,
//     const std::vector<float> &cos_table,
//     const std::vector<float> &sin_table,
//     const std::vector<half> &mask,
//     int block_size_m = 64,
//     int block_size_n = 64) {
//     if (num_q_heads % num_kv_heads != 0) return {};
//     int q_per_kv = num_q_heads / num_kv_heads;
//
//     size_t q_size = static_cast<size_t>(batch) * num_q_heads * m * head_dim;
//     size_t kv_size = static_cast<size_t>(batch) * num_kv_heads * n * head_dim;
//     if (q_mat.size() != q_size || k_mat.size() != kv_size || v_mat.size() != kv_size) {
//         return {};
//     }
//
//     int half_dim = head_dim / 2;
//     if (cos_table.size() < static_cast<size_t>(std::max(m, n)) * half_dim ||
//         sin_table.size() < static_cast<size_t>(std::max(m, n)) * half_dim) {
//         return {};
//     }
//
//     std::vector<float> output(batch * num_q_heads * m * head_dim);
//
//     for (int b = 0; b < batch; ++b) {
//         const T_Q *q_ptr = q_mat.data() + ((b * num_q_heads) * m * head_dim);
//         const T_KV *k_ptr = k_mat.data() + ((b * num_kv_heads) * n * head_dim);
//         const T_KV *v_ptr = v_mat.data() + ((b * num_kv_heads) * n * head_dim);
//         float *out_ptr = output.data() + ((b * num_q_heads) * m * head_dim);
//
//         for (int qh = 0; qh < num_q_heads; ++qh) {
//             int kvh = qh / q_per_kv;
//             flash_attention_cpu_single_head<T_Q, T_KV, rope_flag>(
//                 m, n, head_dim, num_q_heads, num_kv_heads, qh, kvh,
//                 q_ptr, k_ptr, v_ptr,
//                 out_ptr,
//                 cos_table.data(),
//                 sin_table.data(), mask.data(),
//                 block_size_m,
//                 block_size_n
//             );
//         }
//     }
//
//     return output;
// }
//
// static std::vector<float> precompute_rope_tables(int max_seq_len, int dim, float base = 10000.0f) {
//     std::vector<float> inv_freq(dim / 2);
//     for (int i = 0; i < dim / 2; ++i) {
//         inv_freq[i] = 1.0f / std::pow(base, float(2 * i) / dim);
//     }
//
//     std::vector<float> table(max_seq_len * (dim / 2));
//     for (int pos = 0; pos < max_seq_len; ++pos) {
//         for (int i = 0; i < dim / 2; ++i) {
//             float freq = pos * inv_freq[i];
//             table[pos * (dim / 2) + i] = freq;
//         }
//     }
//     return table;
// }
//
// // 然后：
//
// template<typename T_Q, typename T_KV, const int rope_flag>
// void flash_attention_double_buffer_kv(int batch,
//                                       int M, int N, int D,
//                                       int num_q_heads,
//                                       int num_kv_heads,
//                                       const std::vector<T_Q> &q_mat,
//                                       const std::vector<T_KV> &k_mat,
//                                       const std::vector<T_KV> &v_mat,
//                                       const std::vector<float> &cos_table,
//                                       const std::vector<float> &sin_table,
//                                       const std::vector<half> &mask,
//                                       int block_size_m = 64,
//                                       int block_size_n = 64) {
//     T_Q *q_gpu = nullptr;
//     cudaMalloc((void **) &q_gpu, sizeof(T_Q) * batch * num_q_heads * M * D);
//     cudaMemcpy(q_gpu, q_mat.data(), sizeof(T_Q) * batch * num_q_heads * M * D, cudaMemcpyHostToDevice);
//     T_KV *k_gpu = nullptr;
//     cudaMalloc((void **) &k_gpu, sizeof(T_KV) * batch * num_kv_heads * N * D);
//     cudaMemcpy(k_gpu, k_mat.data(), sizeof(T_KV) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
//     T_KV *v_gpu = nullptr;
//     cudaMalloc((void **) &v_gpu, sizeof(T_KV) * batch * num_kv_heads * N * D);
//     cudaMemcpy(v_gpu, v_mat.data(), sizeof(T_KV) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
//     std::vector<float> cos_sin_table(M * D);
//     for (int m = 0; m < M; ++m) {
//         for (int d = 0; d < D; d += 2) {
//             cos_sin_table[m * D + d] = cos_table[m * D / 2 + d / 2];
//             cos_sin_table[m * D + d + 1] = sin_table[m * D / 2 + d / 2];
//             // if (m == 1)
//             //     {
//             //     printf("cpu mm: %d, kk: %d, coff cos: %lf, coff sin: %lf \n", m, d, cos_sin_table[m * D + d], cos_sin_table[m * D + d + 1]);
//             // }
//         }
//         // if (m == 1)
//         //     {
//         //     printf("\n");
//         // }
//     }
//
//     float *cos_sin_table_gpu = nullptr;
//     cudaMalloc((void **) &cos_sin_table_gpu, sizeof(float) * M * D);
//     cudaMemcpy(cos_sin_table_gpu, cos_sin_table.data(), sizeof(float) * M * D, cudaMemcpyHostToDevice);
//
//     T_KV *mask_gpu = nullptr;
//     cudaMalloc((void **) &mask_gpu, sizeof(T_KV) * M * N);
//     cudaMemcpy(mask_gpu, mask.data(), sizeof(T_KV) * M * N, cudaMemcpyHostToDevice);
//
//     float *output = nullptr;
//     cudaMalloc((void **) &output, sizeof(float) * batch * num_q_heads * M * D);
//     cudaMemset(output, 0, sizeof(float) * batch * num_q_heads * M * D);
//     const int block_num = max((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, 1);
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//
//
//     dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, num_q_heads / HEAD_NUM_PER_BLOCK, batch);
//     dim3 block(256);
//     printf("grid x: %d, grid y: %d, grid z: %d \n", grid.x, grid.y, grid.z);
//     printf("block x: %d, block y: %d\n", block.x, block.y);
//
//     flash_attention<T_Q, T_KV, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//                 ELEMENTS_PER_LOAD, rope_flag, B, MBit, S, HEAD_NUM_PER_BLOCK><<<
//             grid
//             , block>>>(
//                 M, N, D,
//                 D, D, D,
//                 (1.0f / std::sqrt(static_cast<float>(D))),
//                 num_q_heads,
//                 num_kv_heads,
//                 q_gpu, k_gpu, v_gpu, mask_gpu, cos_sin_table_gpu,
//                 output);
//     cudaDeviceSynchronize();
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//
//     std::vector<float> output_cpu;
//     output_cpu.resize(batch * num_q_heads * M * D);
//     cudaMemcpy(output_cpu.data(), output, sizeof(float) * batch * num_q_heads * M * D, cudaMemcpyDeviceToHost);
//
//     cudaFree(q_gpu);
//     cudaFree(k_gpu);
//     cudaFree(v_gpu);
//
//     cudaFree(mask_gpu);
//     cudaFree(output);
//     cudaFree(cos_sin_table_gpu);
//
//     if (rope_flag == 1) {
//         const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch
//                                 *
//                                 num_q_heads * M * N + batch * num_q_heads * M * (D / 2) * (4 + 2);
//         const double gflops = static_cast<double>(flops) / 1e9;
//         const double seconds = milliseconds / 1000.0;
//         const double gflops_per_sec = gflops / seconds;
//         printf("******************************************\n");
//         printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
//         printf("Kernel time: %.4f ms\n", milliseconds);
//         printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
//         printf("手写 flash_attention rope double buffer k v Performance: %.2f GFLOPS/s\n", gflops_per_sec);
//     } else {
//         const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch
//                                 *
//                                 num_q_heads * M * N;
//         const double gflops = static_cast<double>(flops) / 1e9;
//         const double seconds = milliseconds / 1000.0;
//         const double gflops_per_sec = gflops / seconds;
//         printf("******************************************\n");
//         printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
//         printf("Kernel time: %.4f ms\n", milliseconds);
//         printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
//         printf("手写 flash_attention Performance: %.2f GFLOPS/s\n", gflops_per_sec);
//     }
//
// #ifdef _DEBUG
//     // printf("q_mat: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("q[%d][%d]: %lf ", i, j, __half2float(q_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // //
//     // printf("k_mat: \n");
//     // for (int i = 0; i < N; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("k[%d][%d]: %lf ", i, j, __half2float(k_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // printf("v_mat: \n");
//     // for (int i = 0; i < N; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(v_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//
//     std::vector<float> cpu_result = flash_attention_cpu_gqa<T_Q, T_KV, rope_flag>(
//         batch, M, N, D, num_q_heads, num_kv_heads, q_mat, k_mat,
//         v_mat, cos_table, sin_table, mask);
//     // printf("cpu_result: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(cpu_result[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // printf("gpu_result: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(output_cpu[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     for (int b = 0; b < batch; b++) {
//         for (int q = 0; q < num_q_heads; q++) {
//             float *cpu_result_single = cpu_result.data() + (b * num_q_heads) * M * D + q * D;
//             float *gpu_result_single = output_cpu.data() + (b * num_q_heads) * M * D + q * D;
//             for (int j = 0; j < M; ++j) {
//                 for (int i = 0; i < D; ++i) {
//                     float delta = cpu_result_single[j * D + i] - gpu_result_single[j * D + i];
//                     if (fabs(delta) > 0.1f) {
//                         printf("error : %f, m: %d, d: %d, cpu: %lf, gpu: %lf\n", delta, j, i,
//                                cpu_result_single[j * D + i], gpu_result_single[j * D + i]);
//                         return;
//                     }
//                 }
//             }
//         }
//     }
//
//     printf("success\n");
// #endif
// }
//
// template<typename T_Q, typename T_KV, const int rope_flag>
// void flash_attention_output_shared(int batch,
//                                    int M, int N, int D,
//                                    int num_q_heads,
//                                    int num_kv_heads,
//                                    const std::vector<T_Q> &q_mat,
//                                    const std::vector<T_KV> &k_mat,
//                                    const std::vector<T_KV> &v_mat,
//                                    const std::vector<float> &cos_table,
//                                    const std::vector<float> &sin_table,
//                                    const std::vector<half> &mask,
//                                    int block_size_m = 64,
//                                    int block_size_n = 64) {
//     T_Q *q_gpu = nullptr;
//     cudaMalloc((void **) &q_gpu, sizeof(T_Q) * batch * num_q_heads * M * D);
//     cudaMemcpy(q_gpu, q_mat.data(), sizeof(T_Q) * batch * num_q_heads * M * D, cudaMemcpyHostToDevice);
//     T_KV *k_gpu = nullptr;
//     cudaMalloc((void **) &k_gpu, sizeof(T_KV) * batch * num_kv_heads * N * D);
//     cudaMemcpy(k_gpu, k_mat.data(), sizeof(T_KV) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
//     T_KV *v_gpu = nullptr;
//     cudaMalloc((void **) &v_gpu, sizeof(T_KV) * batch * num_kv_heads * N * D);
//     cudaMemcpy(v_gpu, v_mat.data(), sizeof(T_KV) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
//     std::vector<float> cos_sin_table(M * D);
//     for (int m = 0; m < M; ++m) {
//         for (int d = 0; d < D; d += 2) {
//             cos_sin_table[m * D + d] = cos_table[m * D / 2 + d / 2];
//             cos_sin_table[m * D + d + 1] = sin_table[m * D / 2 + d / 2];
//             // if (m == 1)
//             //     {
//             //     printf("cpu mm: %d, kk: %d, coff cos: %lf, coff sin: %lf \n", m, d, cos_sin_table[m * D + d], cos_sin_table[m * D + d + 1]);
//             // }
//         }
//         // if (m == 1)
//         //     {
//         //     printf("\n");
//         // }
//     }
//
//     float *cos_sin_table_gpu = nullptr;
//     cudaMalloc((void **) &cos_sin_table_gpu, sizeof(float) * M * D);
//     cudaMemcpy(cos_sin_table_gpu, cos_sin_table.data(), sizeof(float) * M * D, cudaMemcpyHostToDevice);
//
//     T_KV *mask_gpu = nullptr;
//     cudaMalloc((void **) &mask_gpu, sizeof(T_KV) * M * N);
//     cudaMemcpy(mask_gpu, mask.data(), sizeof(T_KV) * M * N, cudaMemcpyHostToDevice);
//
//     float *output = nullptr;
//     cudaMalloc((void **) &output, sizeof(float) * batch * num_q_heads * M * D);
//     cudaMemset(output, 0, sizeof(float) * batch * num_q_heads * M * D);
//     const int block_num = max((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, 1);
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//
//
//     dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, num_q_heads / HEAD_NUM_PER_BLOCK, batch);
//     dim3 block(256);
//     printf("grid x: %d, grid y: %d, grid z: %d \n", grid.x, grid.y, grid.z);
//     printf("block x: %d, block y: %d\n", block.x, block.y);
//
//     flash_attention_double_buffer_k_output_shared<T_Q, T_KV, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N,
//                 BLOCK_DIM_K,
//                 ELEMENTS_PER_LOAD, rope_flag, B, MBit, S, HEAD_NUM_PER_BLOCK><<<
//             grid
//             , block>>>(
//                 M, N, D,
//                 D, D, D,
//                 (1.0f / std::sqrt(static_cast<float>(D))),
//                 num_q_heads,
//                 num_kv_heads,
//                 q_gpu, k_gpu, v_gpu, mask_gpu, cos_sin_table_gpu,
//                 output);
//     cudaDeviceSynchronize();
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//
//     std::vector<float> output_cpu;
//     output_cpu.resize(batch * num_q_heads * M * D);
//     cudaMemcpy(output_cpu.data(), output, sizeof(float) * batch * num_q_heads * M * D, cudaMemcpyDeviceToHost);
//
//     cudaFree(q_gpu);
//     cudaFree(k_gpu);
//     cudaFree(v_gpu);
//
//     cudaFree(mask_gpu);
//     cudaFree(output);
//     cudaFree(cos_sin_table_gpu);
//
//     if (rope_flag == 1) {
//         const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch
//                                 *
//                                 num_q_heads * M * N + batch * num_q_heads * M * (D / 2) * (4 + 2);
//         const double gflops = static_cast<double>(flops) / 1e9;
//         const double seconds = milliseconds / 1000.0;
//         const double gflops_per_sec = gflops / seconds;
//         printf("******************************************\n");
//         printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
//         printf("Kernel time: %.4f ms\n", milliseconds);
//         printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
//         printf("手写 flash_attention rope output shared Performance: %.2f GFLOPS/s\n", gflops_per_sec);
//     } else {
//         const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch
//                                 *
//                                 num_q_heads * M * N;
//         const double gflops = static_cast<double>(flops) / 1e9;
//         const double seconds = milliseconds / 1000.0;
//         const double gflops_per_sec = gflops / seconds;
//         printf("******************************************\n");
//         printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
//         printf("Kernel time: %.4f ms\n", milliseconds);
//         printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
//         printf("手写 flash_attention Performance: %.2f GFLOPS/s\n", gflops_per_sec);
//     }
//
// #ifdef _DEBUG
//     // printf("q_mat: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("q[%d][%d]: %lf ", i, j, __half2float(q_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // //
//     // printf("k_mat: \n");
//     // for (int i = 0; i < N; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("k[%d][%d]: %lf ", i, j, __half2float(k_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // printf("v_mat: \n");
//     // for (int i = 0; i < N; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(v_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//
//     std::vector<float> cpu_result = flash_attention_cpu_gqa<T_Q, T_KV, rope_flag>(
//         batch, M, N, D, num_q_heads, num_kv_heads, q_mat, k_mat,
//         v_mat, cos_table, sin_table, mask);
//     // printf("cpu_result: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(cpu_result[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // printf("gpu_result: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(output_cpu[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     for (int b = 0; b < batch; b++) {
//         for (int q = 0; q < num_q_heads; q++) {
//             float *cpu_result_single = cpu_result.data() + (b * num_q_heads) * M * D + q * D;
//             float *gpu_result_single = output_cpu.data() + (b * num_q_heads) * M * D + q * D;
//             for (int j = 0; j < M; ++j) {
//                 for (int i = 0; i < D; ++i) {
//                     float delta = cpu_result_single[j * D + i] - gpu_result_single[j * D + i];
//                     if (fabs(delta) > 0.1f) {
//                         printf("error : %f, m: %d, d: %d, cpu: %lf, gpu: %lf\n", delta, j, i,
//                                cpu_result_single[j * D + i], gpu_result_single[j * D + i]);
//                         return;
//                     }
//                 }
//             }
//         }
//     }
//
//     printf("success\n");
// #endif
// }
//
// template<typename T_Q, typename T_KV, const int rope_flag>
// void flash_attention_double_buffer_k(int batch,
//                                      int M, int N, int D,
//                                      int num_q_heads,
//                                      int num_kv_heads,
//                                      const std::vector<T_Q> &q_mat,
//                                      const std::vector<T_KV> &k_mat,
//                                      const std::vector<T_KV> &v_mat,
//                                      const std::vector<float> &cos_table,
//                                      const std::vector<float> &sin_table,
//                                      const std::vector<half> &mask,
//                                      int block_size_m = 64,
//                                      int block_size_n = 64) {
//     T_Q *q_gpu = nullptr;
//     cudaMalloc((void **) &q_gpu, sizeof(T_Q) * batch * num_q_heads * M * D);
//     cudaMemcpy(q_gpu, q_mat.data(), sizeof(T_Q) * batch * num_q_heads * M * D, cudaMemcpyHostToDevice);
//     T_KV *k_gpu = nullptr;
//     cudaMalloc((void **) &k_gpu, sizeof(T_KV) * batch * num_kv_heads * N * D);
//     cudaMemcpy(k_gpu, k_mat.data(), sizeof(T_KV) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
//     T_KV *v_gpu = nullptr;
//     cudaMalloc((void **) &v_gpu, sizeof(T_KV) * batch * num_kv_heads * N * D);
//     cudaMemcpy(v_gpu, v_mat.data(), sizeof(T_KV) * batch * num_kv_heads * N * D, cudaMemcpyHostToDevice);
//     std::vector<float> cos_sin_table(M * D);
//     for (int m = 0; m < M; ++m) {
//         for (int d = 0; d < D; d += 2) {
//             cos_sin_table[m * D + d] = cos_table[m * D / 2 + d / 2];
//             cos_sin_table[m * D + d + 1] = sin_table[m * D / 2 + d / 2];
//             // if (m == 1)
//             //     {
//             //     printf("cpu mm: %d, kk: %d, coff cos: %lf, coff sin: %lf \n", m, d, cos_sin_table[m * D + d], cos_sin_table[m * D + d + 1]);
//             // }
//         }
//         // if (m == 1)
//         //     {
//         //     printf("\n");
//         // }
//     }
//
//     float *cos_sin_table_gpu = nullptr;
//     cudaMalloc((void **) &cos_sin_table_gpu, sizeof(float) * M * D);
//     cudaMemcpy(cos_sin_table_gpu, cos_sin_table.data(), sizeof(float) * M * D, cudaMemcpyHostToDevice);
//
//     T_KV *mask_gpu = nullptr;
//     cudaMalloc((void **) &mask_gpu, sizeof(T_KV) * M * N);
//     cudaMemcpy(mask_gpu, mask.data(), sizeof(T_KV) * M * N, cudaMemcpyHostToDevice);
//
//     float *output = nullptr;
//     cudaMalloc((void **) &output, sizeof(float) * batch * num_q_heads * M * D);
//     cudaMemset(output, 0, sizeof(float) * batch * num_q_heads * M * D);
//     const int block_num = max((N + BLOCK_DIM_N - 1) / BLOCK_DIM_N, 1);
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//
//
//     dim3 grid((M + BLOCK_DIM_M - 1) / BLOCK_DIM_M, num_q_heads / HEAD_NUM_PER_BLOCK, batch);
//     dim3 block(256);
//     printf("grid x: %d, grid y: %d, grid z: %d \n", grid.x, grid.y, grid.z);
//     printf("block x: %d, block y: %d\n", block.x, block.y);
//
//     flash_attention_double_buffer_k_out_sm<T_Q, T_KV, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//                 ELEMENTS_PER_LOAD, rope_flag, B, MBit, S, HEAD_NUM_PER_BLOCK, 32, 256><<<
//             grid
//             , block>>>(
//                 M, N, D, std::max(M, N),
//                 D, D, D,
//                 (1.0f / std::sqrt(static_cast<float>(D))),
//                 num_q_heads,
//                 num_kv_heads,
//                 q_gpu, k_gpu, v_gpu, mask_gpu, cos_sin_table_gpu,
//                 output);
//     cudaDeviceSynchronize();
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//
//     std::vector<float> output_cpu;
//     output_cpu.resize(batch * num_q_heads * M * D);
//     cudaMemcpy(output_cpu.data(), output, sizeof(float) * batch * num_q_heads * M * D, cudaMemcpyDeviceToHost);
//
//     cudaFree(q_gpu);
//     cudaFree(k_gpu);
//     cudaFree(v_gpu);
//
//     cudaFree(mask_gpu);
//     cudaFree(output);
//     cudaFree(cos_sin_table_gpu);
//
//     if (rope_flag == 1) {
//         const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch
//                                 *
//                                 num_q_heads * M * N + batch * num_q_heads * M * (D / 2) * (4 + 2);
//         const double gflops = static_cast<double>(flops) / 1e9;
//         const double seconds = milliseconds / 1000.0;
//         const double gflops_per_sec = gflops / seconds;
//         printf("******************************************\n");
//         printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
//         printf("Kernel time: %.4f ms\n", milliseconds);
//         printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
//         printf("手写 flash_attention rope double buffer k Performance: %.2f GFLOPS/s\n", gflops_per_sec);
//     } else {
//         const long long flops = 2ll * batch * num_q_heads * M * N * D + 2 * batch * num_kv_heads * M * N * D + 3 * batch
//                                 *
//                                 num_q_heads * M * N;
//         const double gflops = static_cast<double>(flops) / 1e9;
//         const double seconds = milliseconds / 1000.0;
//         const double gflops_per_sec = gflops / seconds;
//         printf("******************************************\n");
//         printf("Matrix size: %d, %d, %d x %d x %d\n", batch, num_q_heads, M, N, D);
//         printf("Kernel time: %.4f ms\n", milliseconds);
//         printf("FLOPs: %lld (%.2f GFLOPs)\n", flops, gflops);
//         printf("手写 flash_attention Performance: %.2f GFLOPS/s\n", gflops_per_sec);
//     }
//
// #ifdef _DEBUG
//     // printf("q_mat: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("q[%d][%d]: %lf ", i, j, __half2float(q_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // //
//     // printf("k_mat: \n");
//     // for (int i = 0; i < N; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("k[%d][%d]: %lf ", i, j, __half2float(k_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // printf("v_mat: \n");
//     // for (int i = 0; i < N; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(v_mat[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//
//     std::vector<float> cpu_result = flash_attention_cpu_gqa<T_Q, T_KV, rope_flag>(
//         batch, M, N, D, num_q_heads, num_kv_heads, q_mat, k_mat,
//         v_mat, cos_table, sin_table, mask);
//     // printf("cpu_result: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(cpu_result[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     // printf("gpu_result: \n");
//     // for (int i = 0; i < M; i++) {
//     //     for (int j = 0; j < D; j++) {
//     //         printf("v[%d][%d]: %lf ", i, j, __half2float(output_cpu[i * D + j]));
//     //     }
//     //     printf("\n");
//     // }
//     for (int b = 0; b < batch; b++) {
//         for (int q = 0; q < num_q_heads; q++) {
//             float *cpu_result_single = cpu_result.data() + (b * num_q_heads) * M * D + q * D;
//             float *gpu_result_single = output_cpu.data() + (b * num_q_heads) * M * D + q * D;
//             for (int j = 0; j < M; ++j) {
//                 for (int i = 0; i < D; ++i) {
//                     float delta = cpu_result_single[j * D + i] - gpu_result_single[j * D + i];
//                     if (std::isnan(delta) || fabs(delta) > 0.1f) {
//                         printf("error : %f, head_index: %d, m: %d, d: %d, cpu: %lf, gpu: %lf\n", delta, q, j, i,
//                                cpu_result_single[j * D + i], gpu_result_single[j * D + i]);
//                         return;
//                     }
//                 }
//             }
//         }
//     }
//
//     printf("success\n");
// #endif
// }
//
// template<typename T>
// void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
//     for (auto &element: vector) {
//         element = static_cast<T>(dist(mt));
//     }
// }
//
// int main54646(int argc, char *argv) {
//     cudaDeviceProp device_prop{};
//     cudaGetDeviceProperties(&device_prop, 0);
//     printf("device prop sharedMemPerBlock:%d \n", device_prop.sharedMemPerBlock);
//     printf("device prop regsPerBlock: %d\n", device_prop.regsPerBlock);
//     //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//
//     std::mt19937 mt(42);
//     std::uniform_real_distribution<double> dist(-4.0, 4.0);
// #ifdef _DEBUG
//     int dim = 512;
//     int m = dim;
//     int n = dim;
//     int k = HIDDEN_DIM;
//     int num_q_heads = 32;
//     int num_kv_heads = 8;
//     int batch = 1;
//
//
//     std::vector<T_Q> a_mat;
//     a_mat.resize(batch * m * num_q_heads * k);
//     std::vector<T_KV> b_mat;
//     b_mat.resize(batch * n * num_kv_heads * k);
//     std::vector<T_KV> c_mat;
//     c_mat.resize(batch * n * num_kv_heads * k);
//     std::vector<T_KV> mask;
//     mask.resize(m * n);
//     T_KV neg_inf = __float2half(-INFINITY); // 将 -inf 转为 half
//     mask.assign(mask.size(), neg_inf);
//
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j <= i; ++j) {
//             mask[i * n + j] = 0;
//         }
//     }
//
//
//     PopulateVector<T_Q>(a_mat, mt, dist);
//     PopulateVector<T_KV>(b_mat, mt, dist);
//     PopulateVector<T_KV>(c_mat, mt, dist);
//     auto angles = precompute_rope_tables(std::max(m, n), k);
//     std::vector<float> cos_table(angles.size()), sin_table(angles.size());
//     for (size_t i = 0; i < angles.size(); ++i) {
//         cos_table[i] = std::cos(angles[i]);
//         sin_table[i] = std::sin(angles[i]);
//     }
//
//     flash_attention_double_buffer_k<T_Q, T_KV, 1>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat,
//                                                   cos_table, sin_table,
//                                                   mask);
//     // flash_attention_double_buffer_kv<T_Q, T_KV, 1>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat,
//     //                                                cos_table, sin_table,
//     //                                                mask);
//     // flash_attention_output_shared<T_Q, T_KV, 1>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat,
//     //                                             cos_table, sin_table,
//     //                                             mask);
//
// #else
//     int dim = 1024;
//     int m = dim;
//     int n = dim;
//     int k = HIDDEN_DIM;
//     int num_q_heads = 32;
//     int num_kv_heads = 8;
//     int batch = 1;
//
//
//     std::vector<T_Q> a_mat;
//     a_mat.resize(batch * m * num_q_heads * k);
//     std::vector<T_KV> b_mat;
//     b_mat.resize(batch * n * num_kv_heads * k);
//     std::vector<T_KV> c_mat;
//     c_mat.resize(batch * n * num_kv_heads * k);
//     std::vector<T_KV> mask;
//     mask.resize(m * n);
//     T_KV neg_inf = __float2half(-INFINITY); // 将 -inf 转为 half
//     mask.assign(mask.size(), neg_inf);
//
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j <= i; ++j) {
//             mask[i * n + j] = 0;
//         }
//     }
//
//
//     PopulateVector<T_Q>(a_mat, mt, dist);
//     PopulateVector<T_KV>(b_mat, mt, dist);
//     PopulateVector<T_KV>(c_mat, mt, dist);
//     auto angles = precompute_rope_tables(std::max(m, n), k);
//     std::vector<float> cos_table(angles.size()), sin_table(angles.size());
//     for (size_t i = 0; i < angles.size(); ++i) {
//         cos_table[i] = std::cos(angles[i]);
//         sin_table[i] = std::sin(angles[i]);
//     }
//
//     flash_attention_double_buffer_k<T_Q, T_KV, 1>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat,
//                                                   cos_table, sin_table,
//                                                   mask);
//     // flash_attention_double_buffer_kv<T_Q, T_KV, 1>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat,
//     //                                                cos_table, sin_table,
//     //                                                mask);
//     // flash_attention_output_shared<T_Q, T_KV, 1>(batch, m, n, k, num_q_heads, num_kv_heads, a_mat, b_mat, c_mat,
//     //                                             cos_table, sin_table,
//     //                                             mask);
// #endif
//
//
//     return 0;
// }
