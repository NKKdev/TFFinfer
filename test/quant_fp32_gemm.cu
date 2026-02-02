// //
// // Created by nkk on 2026/1/29.
// //
// //
// // Created by nkk on 2026/1/3.
// //
//
// #include <vector>
// #include <random>
// #include "cublas_v2.h"
// #include "mma.h"
// #include <cstdint>
// #include <cstring>
//
// #include "../cmake-build-debug/_deps/fmt-src/include/fmt/compile.h"
// #include "device/cuda/cudaInc.h"
// #include "include/kernel_util.h"
//
// struct Q_8 {
//     static constexpr int BLOCK_SIZE = 32;
//     float d;
//     int8_t qs[BLOCK_SIZE];
//
//     static void dequantize(const Q_8 *blocks, float *out, const int64_t elem_count) {
//         const int nb = elem_count / BLOCK_SIZE;
//         for (int i = 0; i < nb; ++i) {
//             const float scale = __half2float(blocks[i].d);
//             for (int j = 0; j < BLOCK_SIZE; ++j) {
//                 out[i * BLOCK_SIZE + j] = blocks[i].qs[j] * scale;
//             }
//         }
//     }
//
//     //
//     static void quantize(const float *src, Q_8 *blocks, const int64_t elem_count) {
//         const int nb = static_cast<int>(elem_count / BLOCK_SIZE);
//         for (int i = 0; i < nb; ++i) {
//             const float *x = src + i * BLOCK_SIZE;
//
//             float max_abs = 0.0f;
//             for (int j = 0; j < BLOCK_SIZE; ++j) {
//                 max_abs = std::max(max_abs, std::abs(x[j]));
//             }
//             if (max_abs == 0.0f) {
//                 blocks[i].d = tff::utils::fp32_to_fp16(0.0f);
//                 for (int j = 0; j < BLOCK_SIZE; ++j) {
//                     blocks[i].qs[j] = 0;
//                 }
//                 continue;
//             }
//             const float scale = max_abs / 127.0f;
//             const float inv_scale = 1.0f / scale;
//
//             blocks[i].d = tff::utils::fp32_to_fp16(scale);
//
//             for (int j = 0; j < BLOCK_SIZE; ++j) {
//                 const float v = x[j] * inv_scale;
//                 const int32_t iv = static_cast<int32_t>(std::round(v));
//                 blocks[i].qs[j] = static_cast<int8_t>(
//                     std::max(-127, std::min(127, iv))
//                 );
//             }
//         }
//     }
//
//     static constexpr bool is_quantized() { return true; };
// };
//
// static_assert(sizeof(Q_8) == 36);
//
// using T = float;
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
// template<>
// __device__ __forceinline__ void load_vec<int8_t, int32_t>(const int8_t *addr, int32_t *out, const int count) {
//     if (count == 4 && reinterpret_cast<int8_t>(addr) % 16 == 0) {
//         *out = *reinterpret_cast<const int32_t *>(addr);
//     } else {
//         int32_t packed = 0;
// #pragma unroll
//         for (int i = 0; i < count; ++i) {
//             packed |= static_cast<int32_t>(static_cast<uint8_t>(addr[i])) << (i * 8);
//         }
//         out[0] = (packed);
//     }
// }
//
// constexpr int compute_bit(int N) {
//     if (N <= 1) return 0;
//
//     return std::bit_width(static_cast<unsigned>(N - 1));
// }
//
// template<const int B, const int M, const int S>
// __device__ __forceinline__ int swizzle_addr(
//     int row, // 逻辑行号
//     int col // 逻辑列号（以元素为单位）
// ) {
//     int row_low = row & ((1 << B) - 1);
//     int row_high = row >> B;
//
//     int swizzled_col = (row_low << M) ^ col;
//
//     int block_width = 1 << (S + M);
//     int addr = (row_high * block_width) + swizzled_col;
//     return addr;
// }
//
// template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int
//     PAD_SIZE>
// __device__ void load_tile_n(const int ld, const int dim,
//                             const int thread_x, const int thread_y,
//                             const int start_m,
//                             const int k, const int kk,
//                             const T *__restrict__ global_mem,
//                             T *sm) {
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; j++) {
//         int dim0 = start_m + thread_y + j * BLOCK_DIM_LD / VEC_DIM_LD;
//
//         int dim1 = k + thread_x + kk * BLOCK_DIM_K / VEC_DIM_K;
//         T val = 0.0f;
//         if (dim1 < dim && dim0 < ld) {
//             val = __ldg(&global_mem[dim1 * ld + dim0]);
//         }
//         sm[(thread_y + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_LD + PAD_SIZE)
//             + thread_x + kk * BLOCK_DIM_K / VEC_DIM_K] = val;
//     }
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE>
// __device__ void load_tile(const int ld, const int dim,
//                           const int thread_x, const int warp_id,
//                           const int start_block,
//                           const int k,
//                           const Q_8 *__restrict__ global_mem,
//                           int *quant_sm, half *scale_sm) {
//     const int quant_block_id = thread_x / THREAD_NUM_PER_QUANT_BLOCK;
//     const int block_inter_index = thread_x % THREAD_NUM_PER_QUANT_BLOCK;
//
//     //load quant data;
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = start_block + warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD;
// #pragma unroll
//         for (int kk = 0; kk < VEC_DIM_K; ++kk) {
//             int dim1 = k + quant_block_id + kk * QUANT_BLKS_PER_WARP / VEC_DIM_K;
//             if (dim1 < ld && dim0 < dim) {
//                 const Q_8 *val = &global_mem[dim0 * ld + dim1];
//                 int32_t quant_data = 0;
//                 const int8_t *qs_ptr = (&val->qs[0]);
//                 load_vec<int8_t, int32_t>(&qs_ptr[block_inter_index * 4], &quant_data, 4);
//                 quant_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K / sizeof(int) + PAD_SIZE) + thread_x +
//                          kk * (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK] = quant_data;
//             } else {
//                 quant_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K / sizeof(int) + PAD_SIZE) + thread_x +
//                          kk * (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK] = 0;
//             }
//         }
//     }
//     //load scale
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = start_block + warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD;
// #pragma unroll
//         for (int kk = 0; kk < VEC_DIM_K; ++kk) {
//             int dim1 = k + quant_block_id + kk * QUANT_BLKS_PER_WARP / VEC_DIM_K;
//             if (dim1 < ld && dim0 < dim) {
//                 const Q_8 *val = global_mem + dim0 * ld + dim1;
//                 scale_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (QUANT_BLKS_PER_WARP + PAD_SIZE) + quant_block_id +
//                          kk * QUANT_BLKS_PER_WARP / VEC_DIM_K] = val->d;
//             } else {
//                 scale_sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (QUANT_BLKS_PER_WARP + PAD_SIZE) + quant_block_id +
//                          kk * QUANT_BLKS_PER_WARP / VEC_DIM_K] = 0;
//             }
//         }
//     }
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE>
// __device__ void load_tile_double_buffer(const int ld, const int dim,
//                                         const int thread_x, const int warp_id,
//                                         const int start_block,
//                                         const int k, const int kk,
//                                         const void *global_mem,
//                                         int *quant_sm, half *scale_sm) {
//     const int quant_block_id = thread_x / THREAD_NUM_PER_QUANT_BLOCK;
//     const int block_inter_index = thread_x % THREAD_NUM_PER_QUANT_BLOCK;
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int row_stride = BLOCK_DIM_LD / VEC_DIM_LD;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//
//     const int dim0_base = start_block + warp_id;
//
//     const auto *g_ptr = reinterpret_cast<const Q_8 *>(global_mem);
//     //load quant data;
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//         if (dim1 < ld && dim0 < dim) {
//             int32_t quant_data = *reinterpret_cast<const int32_t *>(&g_ptr[dim0 * ld + dim1].qs[block_inter_index * 4]);
//             //load_vec<int8_t, int32_t>(&g_ptr[dim0 * ld + dim1].qs[block_inter_index * 4], &quant_data, 4);
//             quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
//                      kk * quant_col_stride] = quant_data;
//         } else {
//             quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
//                      kk * quant_col_stride] = 0;
//         }
//     }
//
//     //load scale
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//         if (dim1 < ld && dim0 < dim) {
//             const Q_8 *val = &g_ptr[dim0 * ld + dim1];
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = val->d;
//         } else {
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = 0;
//         }
//     }
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE>
// __device__ void load_tile_double_buffer_aligned(const int ld, const int dim,
//                                                 const int thread_x, const int warp_id,
//                                                 const int start_block,
//                                                 const int k, const int kk,
//                                                 const void *global_mem,
//                                                 int *quant_sm, half *scale_sm) {
//     const int quant_block_id = thread_x / THREAD_NUM_PER_QUANT_BLOCK;
//     const int block_inter_index = thread_x % THREAD_NUM_PER_QUANT_BLOCK;
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int row_stride = BLOCK_DIM_LD / VEC_DIM_LD;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//
//     const int dim0_base = start_block + warp_id;
//     const auto *g_ptr = reinterpret_cast<const Q_8 *>(global_mem);
//     //load quant data;
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//
//         if (dim1 < ld && dim0 < dim) {
//             const Q_8 *val = &g_ptr[dim0 * ld + dim1];
//             const int *quant_data = reinterpret_cast<const int *>(&val->qs[0]);
//             quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
//                      kk * quant_col_stride] = quant_data[block_inter_index];
//         } else {
//             quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
//                      kk * quant_col_stride] = 0;
//         }
//     }
//
//     //load scale
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//         if (dim1 < ld && dim0 < dim) {
//             const Q_8 *val = g_ptr + dim0 * ld + dim1;
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = __float2half(val->d);
//         } else {
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = 0;
//         }
//     }
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE>
// __device__ void load_tile_double_buffer_vec(const int ld, const int dim,
//                                             const int thread_x, const int warp_id,
//                                             const int start_block,
//                                             const int k, const int kk,
//                                             const Q_8 *__restrict__ global_mem,
//                                             int4 *quant_sm, half *scale_sm) {
//     const int quant_block_id = thread_x / THREAD_NUM_PER_QUANT_BLOCK;
//     const int block_inter_index = thread_x % THREAD_NUM_PER_QUANT_BLOCK;
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int4) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int row_stride = BLOCK_DIM_LD / VEC_DIM_LD;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//     //int4 *quant_sm_ptr = reinterpret_cast<int4*>(quant_sm);
//     const int dim0_base = start_block + warp_id;
//     //load quant data;
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//         if (dim1 < ld && dim0 < dim) {
//             const Q_8 *val = &global_mem[dim0 * ld + dim1];
//             const int4 *quant_data = reinterpret_cast<const int4 *>(&val->qs[0]);
//             quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
//                      kk * quant_col_stride] = quant_data[block_inter_index];
//         } else {
//             quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
//                      kk * quant_col_stride] = make_int4(0, 0, 0, 0);
//         }
//     }
//
//     //load scale
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//         if (dim1 < ld && dim0 < dim) {
//             const Q_8 *val = global_mem + dim0 * ld + dim1;
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = val->d;
//         } else {
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = 0;
//         }
//     }
// }
//
// template<const int VEC_DIM_LD, const int VEC_DIM_K, const int BLOCK_DIM_LD, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
//     const int B, const int M, const int S>
// __device__ void load_tile_double_buffer_swizzle(const int ld, const int dim,
//                                                 const int thread_x, const int warp_id,
//                                                 const int start_block,
//                                                 const int k, const int kk,
//                                                 const Q_8 *__restrict__ global_mem,
//                                                 int *quant_sm, half *scale_sm) {
//     const int quant_block_id = thread_x / THREAD_NUM_PER_QUANT_BLOCK;
//     const int block_inter_index = thread_x % THREAD_NUM_PER_QUANT_BLOCK;
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int row_stride = BLOCK_DIM_LD / VEC_DIM_LD;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//
//     const int dim0_base = start_block + warp_id;
//     //load quant data;
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//         if (dim1 < ld && dim0 < dim) {
//             const Q_8 *val = &global_mem[dim0 * ld + dim1];
//             const int *quant_data = reinterpret_cast<const int *>(&val->qs[0]);
//             // quant_sm[(warp_id + j * row_stride) * quant_col_width + thread_x +
//             //          kk * quant_col_stride] = quant_data[block_inter_index];
//             int addr = swizzle_addr<B, M, S>((warp_id + j * row_stride), thread_x +
//                                                                          kk * quant_col_stride);
//             quant_sm[addr] = quant_data[block_inter_index];
//         } else {
//             int addr = swizzle_addr<B, M, S>((warp_id + j * row_stride), thread_x +
//                                                                          kk * quant_col_stride);
//             quant_sm[addr / sizeof(int)] = 0;
//         }
//     }
//
//     //load scale
// #pragma unroll
//     for (int j = 0; j < VEC_DIM_LD; ++j) {
//         int dim0 = dim0_base + j * row_stride;
//
//         int dim1 = k + quant_block_id;
//         if (dim1 < ld && dim0 < dim) {
//             const Q_8 *val = global_mem + dim0 * ld + dim1;
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = val->d;
//         } else {
//             scale_sm[(warp_id + j * row_stride) * scale_col_width + quant_block_id +
//                      kk * scale_col_stride] = 0;
//         }
//     }
// }
//
// static __device__ int vec_dot_product(int a, int b, int c_sum) {
//     return __dp4a(a, b, c_sum);
// }
//
// template<const int VEC_DIM_M, const int VEC_DIM_N,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
// __device__ void compute_tile(const int thread_x, const int warp_id,
//                              const int *quant_a_sm, half *a_scale_sm,
//                              const int *quant_b_sm, half *b_scale_sm,
//                              float *c_reg) {
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int m_row_stride = BLOCK_DIM_M / VEC_DIM_M;
//     const int n_row_stride = BLOCK_DIM_N / VEC_DIM_N;
//     const int4 *quant_a_sm_ptr = reinterpret_cast<const int4 *>(quant_a_sm);
//     const int4 *quant_b_sm_ptr = reinterpret_cast<const int4 *>(quant_b_sm);
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             float sum = 0;
// #pragma unroll
//             for (int kk = 0; kk < BLOCK_DIM_K / QUANT_BLOCK_SIZE; kk++) {
//                 float a_scale[VEC_DIM_M];
//                 float b_scale[VEC_DIM_N];
//                 a_scale[mm] = __half2float(
//                     a_scale_sm[(warp_id + mm * m_row_stride) * scale_col_width + kk]);
//                 b_scale[nn] = __half2float(
//                     b_scale_sm[(thread_x + nn * n_row_stride) * scale_col_width + kk]);
//                 int block_sum = 0;
// #pragma unroll
//                 for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q += 4) {
//                     const int kk_index = kk * VEC_DOT_PRODUCT + kk_q;
//                     int4 a_reg[VEC_DIM_M] = {make_int4(0, 0, 0, 0)};
//                     int4 b_reg[VEC_DIM_N] = {make_int4(0, 0, 0, 0)};
//
//                     a_reg[mm] = quant_a_sm_ptr[(warp_id + mm * m_row_stride) * quant_col_width / 4 + kk_index / 4];
//                     b_reg[nn] = quant_b_sm_ptr[(thread_x + nn * n_row_stride) * quant_col_width / 4 + kk_index / 4];
//
//                     block_sum = vec_dot_product(a_reg[mm].x, b_reg[nn].x, block_sum);
//                     block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
//                     block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
//                     block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
//                 }
//                 // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                 //     printf("kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf \n",kk, thread_x, warp_id, block_sum,mm, a_scale[mm], nn,b_scale[nn]);
//                 // }
//                 sum += (block_sum) * (a_scale[mm]) * (b_scale[nn]);
//                 // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                 //     printf("kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,sum:%lf \n",kk, thread_x, warp_id, block_sum,mm, a_scale[mm], nn,b_scale[nn], sum);
//                 // }
//             }
//
//             c_reg[mm * VEC_DIM_N + nn] += sum;
//         }
//     }
// }
//
// template<const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT, const int THREAD_NUM_PER_QUANT_BLOCK>
// __device__ void compute_tile_double_buffer(const int k_size, const int thread_x, const int warp_id,
//                                            const int k_block_index,
//                                            const int *quant_a_sm, half *a_scale_sm,
//                                            const int *quant_b_sm, half *b_scale_sm,
//                                            float *c_reg) {
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int m_row_stride = BLOCK_DIM_M / VEC_DIM_M;
//     const int n_row_stride = BLOCK_DIM_N / VEC_DIM_N;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//     const int4 *quant_a_sm_ptr = reinterpret_cast<const int4 *>(quant_a_sm);
//     const int4 *quant_b_sm_ptr = reinterpret_cast<const int4 *>(quant_b_sm);
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             float sum = 0;
//             //#pragma unroll
//             for (int kk = 0; kk < k_size; kk++) {
//                 float a_scale;
//                 float b_scale;
//                 a_scale = __half2float(
//                     a_scale_sm[(warp_id + mm * m_row_stride) * scale_col_width + kk +
//                                k_block_index * scale_col_stride]);
//                 b_scale = __half2float(
//                     b_scale_sm[(thread_x + nn * n_row_stride) * scale_col_width + kk +
//                                k_block_index * scale_col_stride]);
//                 int block_sum = 0;
// #pragma unroll
//                 for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q += 4) {
//                     const int kk_index = kk * VEC_DOT_PRODUCT + kk_q + k_block_index * quant_col_stride;
//                     int4 a_reg = {make_int4(0, 0, 0, 0)};
//                     int4 b_reg = {make_int4(0, 0, 0, 0)};
//                     a_reg = quant_a_sm_ptr[(warp_id + mm * m_row_stride) * quant_col_width / 4 + kk_index / 4];
//                     b_reg = quant_b_sm_ptr[(thread_x + nn * n_row_stride) * quant_col_width / 4 + kk_index / 4];
//
//                     block_sum = vec_dot_product(a_reg.x, b_reg.x, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.x, nn, b_reg.x,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg.y, b_reg.y, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.y, nn, b_reg.y,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg.z, b_reg.z, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.z, nn, b_reg.z,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg.w, b_reg.w, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.w, nn, b_reg.w,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].w, block_sum);
//                     //
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].w, block_sum);
//                     //
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
//
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].x, nn, b_reg[nn].x,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].y, nn, b_reg[nn].y,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].z, nn, b_reg[nn].z,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].w, nn, b_reg[nn].w,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     // }
//                 }
//
//                 //sum = fmaf(fmaf(block_sum, a_scale, 0.0f), b_scale,sum);
//                 sum += static_cast<float>(block_sum) * a_scale * b_scale;
//                 // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                 //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_scale: %lf, b_scale: %lf ,block_sum: %d, sum: %lf\n",
//                 //         k_block_index, thread_x, warp_id, a_scale,b_scale, block_sum, sum);
//                 // }
//             }
//
//             c_reg[mm * VEC_DIM_N + nn] += sum;
//         }
//     }
// }
// template<const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT, const int THREAD_NUM_PER_QUANT_BLOCK>
// __device__ void compute_tile_double_buffer_fp32(const int k_size, const int thread_x, const int warp_id,
//                                            const int k_block_index,
//                                            const int *quant_a_sm, half *a_scale_sm,
//                                            float *b_sm,
//                                            float *c_reg) {
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int m_row_stride = BLOCK_DIM_M / VEC_DIM_M;
//     const int n_row_stride = BLOCK_DIM_N / VEC_DIM_N;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//     const int4 *quant_a_sm_ptr = reinterpret_cast<const int4 *>(quant_a_sm);
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             float sum = 0;
//             //#pragma unroll
//             for (int kk = 0; kk < k_size; kk++) {
//                 float a_scale = __half2float(
//                     a_scale_sm[(warp_id + mm * m_row_stride) * scale_col_width + kk +
//                                k_block_index * scale_col_stride]);
//                 int block_sum = 0;
// #pragma unroll
//                 for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q += 4) {
//                     const int kk_index = kk * VEC_DOT_PRODUCT + kk_q + k_block_index * quant_col_stride;
//                     int4 a_reg = {make_int4(0, 0, 0, 0)};
//                     int4 b_reg = {make_int4(0, 0, 0, 0)};
//                     a_reg = quant_a_sm_ptr[(warp_id + mm * m_row_stride) * quant_col_width / 4 + kk_index / 4];
//                     b_reg = b_sm[(thread_x + nn * n_row_stride) * quant_col_width / 4 + kk_index / 4];
//
//                     block_sum = vec_dot_product(a_reg.x, b_reg.x, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.x, nn, b_reg.x,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg.y, b_reg.y, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.y, nn, b_reg.y,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg.z, b_reg.z, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.z, nn, b_reg.z,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg.w, b_reg.w, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg.w, nn, b_reg.w,mm, a_scale, nn,b_scale, block_sum);
//                     // }
//
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].w, block_sum);
//                     //
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].w, block_sum);
//                     //
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
//
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].x, nn, b_reg[nn].x,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].y, nn, b_reg[nn].y,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].z, nn, b_reg[nn].z,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].w, nn, b_reg[nn].w,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     // }
//                 }
//
//                 //sum = fmaf(fmaf(block_sum, a_scale, 0.0f), b_scale,sum);
//                 sum += static_cast<float>(block_sum) * a_scale * b_scale;
//                 // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                 //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_scale: %lf, b_scale: %lf ,block_sum: %d, sum: %lf\n",
//                 //         k_block_index, thread_x, warp_id, a_scale,b_scale, block_sum, sum);
//                 // }
//             }
//
//             c_reg[mm * VEC_DIM_N + nn] += sum;
//         }
//     }
// }
// template<const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT, const int THREAD_NUM_PER_QUANT_BLOCK>
// __device__ void compute_tile_double_buffer_vec(const int k_size, const int thread_x, const int warp_id,
//                                                const int k_block_index,
//                                                const int4 *quant_a_sm, half *a_scale_sm,
//                                                const int4 *quant_b_sm, half *b_scale_sm,
//                                                float *c_reg) {
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int4) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int m_row_stride = BLOCK_DIM_M / VEC_DIM_M;
//     const int n_row_stride = BLOCK_DIM_N / VEC_DIM_N;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             float sum = 0;
// #pragma unroll
//             for (int kk = 0; kk < k_size; kk++) {
//                 float a_scale[VEC_DIM_M];
//                 float b_scale[VEC_DIM_N];
//                 a_scale[mm] = __half2float(
//                     a_scale_sm[(warp_id + mm * m_row_stride) * scale_col_width + kk +
//                                k_block_index * scale_col_stride]);
//
//                 b_scale[nn] = __half2float(
//                     b_scale_sm[(thread_x + nn * n_row_stride) * scale_col_width + kk +
//                                k_block_index * scale_col_stride]);
//                 int block_sum = 0;
// #pragma unroll
//                 for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q++) {
//                     const int kk_index = kk * VEC_DOT_PRODUCT + kk_q + k_block_index * quant_col_stride;
//                     int4 a_reg[VEC_DIM_M] = {make_int4(0, 0, 0, 0)};
//                     int4 b_reg[VEC_DIM_N] = {make_int4(0, 0, 0, 0)};
//                     a_reg[mm] = quant_a_sm[(warp_id + mm * m_row_stride) * quant_col_width + kk_index];
//                     b_reg[nn] = quant_b_sm[(thread_x + nn * n_row_stride) * quant_col_width + kk_index];
//                     //a_reg[mm] = quant_a_sm[0];
//                     block_sum = vec_dot_product(a_reg[mm].x, b_reg[nn].x, block_sum);
//                     if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                         printf(
//                             "k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                             k_block_index, thread_x, warp_id, mm, a_reg[mm].x, nn, b_reg[nn].x, mm, a_scale[mm], nn,
//                             b_scale[nn], block_sum);
//                     }
//                     block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
//                     if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                         printf(
//                             "k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                             k_block_index, thread_x, warp_id, mm, a_reg[mm].y, nn, b_reg[nn].y, mm, a_scale[mm], nn,
//                             b_scale[nn], block_sum);
//                     }
//                     block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
//                     if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                         printf(
//                             "k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                             k_block_index, thread_x, warp_id, mm, a_reg[mm].z, nn, b_reg[nn].z, mm, a_scale[mm], nn,
//                             b_scale[nn], block_sum);
//                     }
//                     block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
//                     if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                         printf(
//                             "k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                             k_block_index, thread_x, warp_id, mm, a_reg[mm].w, nn, b_reg[nn].w, mm, a_scale[mm], nn,
//                             b_scale[nn], block_sum);
//                     }
//                 }
//                 if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     printf(
//                         "k_block_index: %d, kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf \n",
//                         k_block_index, kk, thread_x, warp_id, block_sum, mm, a_scale[mm], nn, b_scale[nn]);
//                 }
//                 sum += (block_sum) * (a_scale[mm]) * (b_scale[nn]);
//                 if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 0) {
//                     printf(
//                         "k_block_index: %d, kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,sum:%lf \n",
//                         k_block_index, kk, thread_x, warp_id, block_sum, mm, a_scale[mm], nn, b_scale[nn], sum);
//                 }
//             }
//
//             c_reg[mm * VEC_DIM_N + nn] += sum;
//         }
//     }
// }
//
// template<const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT, const int THREAD_NUM_PER_QUANT_BLOCK,
//     const int B, const int M, const int S>
// __device__ void compute_tile_double_buffer_swizzle(const int k_size, const int thread_x, const int warp_id,
//                                                    const int k_block_index,
//                                                    const int *quant_a_sm, half *a_scale_sm,
//                                                    const int *quant_b_sm, half *b_scale_sm,
//                                                    float *c_reg) {
//     const int quant_col_width = (BLOCK_DIM_K / sizeof(int) + PAD_SIZE);
//     const int scale_col_width = (QUANT_BLKS_PER_WARP + PAD_SIZE);
//     const int m_row_stride = BLOCK_DIM_M / VEC_DIM_M;
//     const int n_row_stride = BLOCK_DIM_N / VEC_DIM_N;
//     const int quant_col_stride = (QUANT_BLKS_PER_WARP / VEC_DIM_K) * THREAD_NUM_PER_QUANT_BLOCK;
//     const int scale_col_stride = QUANT_BLKS_PER_WARP / VEC_DIM_K;
//     const int4 *quant_a_sm_ptr = reinterpret_cast<const int4 *>(quant_a_sm);
//     const int4 *quant_b_sm_ptr = reinterpret_cast<const int4 *>(quant_b_sm);
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             float sum = 0;
// #pragma unroll
//             for (int kk = 0; kk < k_size; kk++) {
//                 float a_scale[VEC_DIM_M];
//                 float b_scale[VEC_DIM_N];
//
//                 a_scale[mm] = __half2float(
//                     a_scale_sm[(warp_id + mm * m_row_stride) * scale_col_width + kk +
//                                k_block_index * scale_col_stride]);
//                 b_scale[nn] = __half2float(
//                     b_scale_sm[(thread_x + nn * n_row_stride) * scale_col_width + kk +
//                                k_block_index * scale_col_stride]);
//                 int block_sum = 0;
// #pragma unroll
//                 for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q += 4) {
//                     const int kk_index = kk * VEC_DOT_PRODUCT + kk_q + k_block_index * quant_col_stride;
//                     int4 a_reg[VEC_DIM_M] = {make_int4(0, 0, 0, 0)};
//                     int4 b_reg[VEC_DIM_N] = {make_int4(0, 0, 0, 0)};
//                     int a_addr = swizzle_addr<B, M, S>(warp_id + mm * m_row_stride, kk_index);
//                     a_reg[mm] = quant_a_sm_ptr[a_addr / 4];
//                     int b_addr = swizzle_addr<B, M, S>((thread_x + nn * n_row_stride), kk_index);
//                     b_reg[nn] = quant_b_sm_ptr[b_addr / 4];
//
//                     block_sum = vec_dot_product(a_reg[mm].x, b_reg[nn].x, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].x, nn, b_reg[nn].x,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].y, nn, b_reg[nn].y,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].z, nn, b_reg[nn].z,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     // }
//                     block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].w, nn, b_reg[nn].w,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     // }
//
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].y, b_reg[nn].w, block_sum);
//                     //
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].z, b_reg[nn].w, block_sum);
//                     //
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].x, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].y, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].z, block_sum);
//                     // block_sum = vec_dot_product(a_reg[mm].w, b_reg[nn].w, block_sum);
//
//                     // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].x, nn, b_reg[nn].x,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].y, nn, b_reg[nn].y,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].z, nn, b_reg[nn].z,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     //     printf("k_block_index: %d, thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,block_sum: %d\n",
//                     //         k_block_index, thread_x, warp_id, mm, a_reg[mm].w, nn, b_reg[nn].w,mm, a_scale[mm], nn,b_scale[nn], block_sum);
//                     // }
//                 }
//                 // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                 //     printf("k_block_index: %d, kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf \n",
//                 //         k_block_index, kk, thread_x, warp_id, block_sum,mm, a_scale[mm], nn,b_scale[nn]);
//                 // }
//                 sum += (block_sum) * (a_scale[mm]) * (b_scale[nn]);
//                 // if (blockIdx.y == 0 && blockIdx.x == 0 && warp_id == 0 && mm == 0 && nn == 0 && thread_x == 1) {
//                 //     printf("k_block_index: %d, kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,sum:%lf \n",
//                 //         k_block_index, kk, thread_x, warp_id, block_sum,mm, a_scale[mm], nn,b_scale[nn], sum);
//                 // }
//             }
//
//             c_reg[mm * VEC_DIM_N + nn] += sum;
//         }
//     }
// }
//
// template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
// __global__ void gemm_quant_q_8_0_nt(
//     int M, int N, int K,
//     int a_ld, int b_ld, int c_ld,
//     const Q_8 *__restrict__ a,
//     const Q_8 *__restrict__ b,
//     T *__restrict__ c) {
//     const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//
//     const int warp_id = g_thread_id / (32);
//     const int thread_x = g_thread_id % (32);
//
//     const int start_m = blockIdx.y * BLOCK_DIM_M;
//     const int start_n = blockIdx.x * BLOCK_DIM_N;
//
//     __shared__ int a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//     __shared__ int b_quant_data_sm[BLOCK_DIM_N][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//     __shared__ half a_scale_data_sm[BLOCK_DIM_M][QUANT_BLKS_PER_WARP + PAD_SIZE];
//     __shared__ half b_scale_data_sm[BLOCK_DIM_N][QUANT_BLKS_PER_WARP + PAD_SIZE];
//
//     float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};
//
//     for (size_t k = 0; k < K; k += BLOCK_DIM_K / QUANT_BLOCK_SIZE) {
//         load_tile<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK,
//             PAD_SIZE>(
//             a_ld, M, thread_x, warp_id, start_m, k, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
//         load_tile<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK,
//             PAD_SIZE>(
//             b_ld, N, thread_x, warp_id, start_n, k, b, &b_quant_data_sm[0][0], &b_scale_data_sm[0][0]);
//         __syncthreads();
//
//         compute_tile<VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP, PAD_SIZE,
//             QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT>(
//             thread_x, warp_id,
//             &a_quant_data_sm[0][0], &a_scale_data_sm[0][0],
//             &b_quant_data_sm[0][0], &b_scale_data_sm[0][0],
//             &c_reg[0][0]);
//
//         __syncthreads();
//     }
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const int dst_m_index = start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
//         if (dst_m_index >= M) {
//             continue;
//         }
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             const int dst_n_index = start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N;
//             if (dst_n_index >= N) {
//                 continue;
//             }
//             if (std::is_same_v<T, half>) {
//                 half old_value = c[dst_m_index * c_ld + dst_n_index];
//                 c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
//             } else if (std::is_same_v<T, float>) {
//                 c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
//             }
//         }
//     }
// }
//
// template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
// __global__ void gemm_quant_q_8_0_fp32_nt_double_buffer(
//     int M, int N, int K,
//     int a_ld, int b_ld, int c_ld,
//     const Q_8 *__restrict__ a,
//     const float *__restrict__ b,
//     T *__restrict__ c) {
//     const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//
//     const int warp_id = g_thread_id / (32);
//     const int thread_x = g_thread_id % (32);
//
//     const int start_m = blockIdx.y * BLOCK_DIM_M;
//     const int start_n = blockIdx.x * BLOCK_DIM_N;
//
//
//     __shared__ int a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//     __shared__ half a_scale_data_sm[BLOCK_DIM_M][QUANT_BLKS_PER_WARP + PAD_SIZE];
//
//     __shared__ float b_sm[BLOCK_DIM_N][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//
//
//     float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};
//     int flip_flag = 0;
//
//     load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//         THREAD_NUM_PER_QUANT_BLOCK,
//         PAD_SIZE>(
//         a_ld, M, thread_x, warp_id, start_m, 0, flip_flag, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
//
//     load_tile_n<float, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE>(
//         b_ld, N, thread_x, warp_id, start_n, 0, flip_flag, b, &b_sm[0][0]);
//     __syncthreads();
//
//     for (int k = 0; k <= K; k += ((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K)) {
//         const int k_size = min((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K, K - k);
//         compute_tile_double_buffer<VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//             QUANT_BLKS_PER_WARP,
//             PAD_SIZE,
//             QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT, THREAD_NUM_PER_QUANT_BLOCK>(
//             k_size, thread_x, warp_id, flip_flag,
//             &a_quant_data_sm[0][0], &a_scale_data_sm[0][0],
//             &b_sm[0][0],
//             &c_reg[0][0]);
//
//         const int next_k = k + (BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K;
//         if (next_k < K) {
//             //printf("error:  next_k: %d,flip_flag: %d !flip_flag: %d \n", next_k, flip_flag, !flip_flag);
//             load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//                 THREAD_NUM_PER_QUANT_BLOCK,
//                 PAD_SIZE>(
//                 a_ld, M, thread_x, warp_id, start_m, next_k, !flip_flag, a, &a_quant_data_sm[0][0],
//                 &a_scale_data_sm[0][0]);
//             load_tile_n<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//                 THREAD_NUM_PER_QUANT_BLOCK,
//                 PAD_SIZE>(
//                 b_ld, N, thread_x, warp_id, start_n, next_k, !flip_flag, b, &b_sm[0][0]);
//         }
//         __syncthreads();
//         flip_flag ^= 1;
//     }
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const int dst_m_index = start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
//         if (dst_m_index >= M) {
//             continue;
//         }
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             const int dst_n_index = start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N;
//             if (dst_n_index >= N) {
//                 continue;
//             }
//             if (std::is_same_v<T, half>) {
//                 half old_value = c[dst_m_index * c_ld + dst_n_index];
//                 c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
//             } else if (std::is_same_v<T, float>) {
//                 c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
//             }
//         }
//     }
// }
//
// template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
// __global__ void gemm_quant_q_8_0_nt_double_buffer(
//     int M, int N, int K,
//     int a_ld, int b_ld, int c_ld,
//     const void *a,
//     const void *b,
//     T *__restrict__ c) {
//     const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//
//     const int warp_id = g_thread_id / (32);
//     const int thread_x = g_thread_id % (32);
//
//     const int start_m = blockIdx.y * BLOCK_DIM_M;
//     const int start_n = blockIdx.x * BLOCK_DIM_N;
//
//     __shared__ int a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//     __shared__ int b_quant_data_sm[BLOCK_DIM_N][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//     __shared__ half a_scale_data_sm[BLOCK_DIM_M][QUANT_BLKS_PER_WARP + PAD_SIZE];
//     __shared__ half b_scale_data_sm[BLOCK_DIM_N][QUANT_BLKS_PER_WARP + PAD_SIZE];
//
//     float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};
//     int flip_flag = 0;
//
//     load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//         THREAD_NUM_PER_QUANT_BLOCK,
//         PAD_SIZE>(
//         a_ld, M, thread_x, warp_id, start_m, 0, flip_flag, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
//     load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//         THREAD_NUM_PER_QUANT_BLOCK,
//         PAD_SIZE>(
//         b_ld, N, thread_x, warp_id, start_n, 0, flip_flag, b, &b_quant_data_sm[0][0], &b_scale_data_sm[0][0]);
//     __syncthreads();
//
//     for (int k = 0; k <= K; k += ((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K)) {
//         const int k_size = min((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K, K - k);
//         compute_tile_double_buffer<VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//             QUANT_BLKS_PER_WARP,
//             PAD_SIZE,
//             QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT, THREAD_NUM_PER_QUANT_BLOCK>(
//             k_size, thread_x, warp_id, flip_flag,
//             &a_quant_data_sm[0][0], &a_scale_data_sm[0][0],
//             &b_quant_data_sm[0][0], &b_scale_data_sm[0][0],
//             &c_reg[0][0]);
//
//         const int next_k = k + (BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K;
//         if (next_k < K) {
//             //printf("error:  next_k: %d,flip_flag: %d !flip_flag: %d \n", next_k, flip_flag, !flip_flag);
//             load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//                 THREAD_NUM_PER_QUANT_BLOCK,
//                 PAD_SIZE>(
//                 a_ld, M, thread_x, warp_id, start_m, next_k, !flip_flag, a, &a_quant_data_sm[0][0],
//                 &a_scale_data_sm[0][0]);
//             load_tile_double_buffer<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//                 THREAD_NUM_PER_QUANT_BLOCK,
//                 PAD_SIZE>(
//                 b_ld, N, thread_x, warp_id, start_n, next_k, !flip_flag, b, &b_quant_data_sm[0][0],
//                 &b_scale_data_sm[0][0]);
//         }
//         __syncthreads();
//         flip_flag ^= 1;
//     }
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const int dst_m_index = start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
//         if (dst_m_index >= M) {
//             continue;
//         }
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             const int dst_n_index = start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N;
//             if (dst_n_index >= N) {
//                 continue;
//             }
//             if (std::is_same_v<T, half>) {
//                 half old_value = c[dst_m_index * c_ld + dst_n_index];
//                 c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
//             } else if (std::is_same_v<T, float>) {
//                 c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
//             }
//         }
//     }
// }
//
// template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
// __global__ void gemm_quant_q_8_0_nt_double_buffer_vec(
//     int M, int N, int K,
//     int a_ld, int b_ld, int c_ld,
//     const Q_8 *__restrict__ a,
//     const Q_8 *__restrict__ b,
//     T *__restrict__ c) {
//     const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//
//     const int warp_id = g_thread_id / (32);
//     const int thread_x = g_thread_id % (32);
//
//     const int start_m = blockIdx.y * BLOCK_DIM_M;
//     const int start_n = blockIdx.x * BLOCK_DIM_N;
//
//     __shared__ int4 a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K / sizeof(int4) + PAD_SIZE];
//     __shared__ int4 b_quant_data_sm[BLOCK_DIM_N][BLOCK_DIM_K / sizeof(int4) + PAD_SIZE];
//     __shared__ half a_scale_data_sm[BLOCK_DIM_M][QUANT_BLKS_PER_WARP + PAD_SIZE];
//     __shared__ half b_scale_data_sm[BLOCK_DIM_N][QUANT_BLKS_PER_WARP + PAD_SIZE];
//
//     float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};
//     int flip_flag = 0;
//
//     load_tile_double_buffer_vec<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//         THREAD_NUM_PER_QUANT_BLOCK,
//         PAD_SIZE>(
//         a_ld, M, thread_x, warp_id, start_m, 0, flip_flag, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
//     load_tile_double_buffer_vec<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//         THREAD_NUM_PER_QUANT_BLOCK,
//         PAD_SIZE>(
//         b_ld, N, thread_x, warp_id, start_n, 0, flip_flag, b, &b_quant_data_sm[0][0], &b_scale_data_sm[0][0]);
//     __syncthreads();
//
//     for (int k = 0; k <= K; k += ((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K)) {
//         const int k_size = min((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K, K - k);
//         compute_tile_double_buffer_vec<VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//             QUANT_BLKS_PER_WARP,
//             PAD_SIZE,
//             QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT, THREAD_NUM_PER_QUANT_BLOCK>(
//             k_size, thread_x, warp_id, flip_flag,
//             &a_quant_data_sm[0][0], &a_scale_data_sm[0][0],
//             &b_quant_data_sm[0][0], &b_scale_data_sm[0][0],
//             &c_reg[0][0]);
//
//         const int next_k = k + (BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K;
//         if (next_k < K) {
//             //printf("error:  next_k: %d,flip_flag: %d !flip_flag: %d \n", next_k, flip_flag, !flip_flag);
//             // load_tile_double_buffer_vec<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//             //     THREAD_NUM_PER_QUANT_BLOCK,
//             //     PAD_SIZE>(
//             //     a_ld, M, thread_x, warp_id, start_m, next_k, !flip_flag, a, &a_quant_data_sm[0][0],
//             //     &a_scale_data_sm[0][0]);
//             // load_tile_double_buffer_vec<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//             //     THREAD_NUM_PER_QUANT_BLOCK,
//             //     PAD_SIZE>(
//             //     b_ld, N, thread_x, warp_id, start_n, next_k, !flip_flag, b, &b_quant_data_sm[0][0],
//             //     &b_scale_data_sm[0][0]);
//         }
//         __syncthreads();
//         flip_flag ^= 1;
//     }
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const int dst_m_index = start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
//         if (dst_m_index >= M) {
//             continue;
//         }
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             const int dst_n_index = start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N;
//             if (dst_n_index >= N) {
//                 continue;
//             }
//             if (std::is_same_v<T, half>) {
//                 half old_value = c[dst_m_index * c_ld + dst_n_index];
//                 c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
//             } else if (std::is_same_v<T, float>) {
//                 c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
//             }
//         }
//     }
// }
//
// template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int VEC_DIM_K,
//     const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K,
//     const int QUANT_BLKS_PER_WARP, const int THREAD_NUM_PER_QUANT_BLOCK, const int PAD_SIZE,
//     const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT,
//     const int B, const int G, const int S>
// __global__ void gemm_quant_q_8_0_nt_double_buffer_swizzle(
//     int M, int N, int K,
//     int a_ld, int b_ld, int c_ld,
//     const Q_8 *__restrict__ a,
//     const Q_8 *__restrict__ b,
//     T *__restrict__ c) {
//     const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//
//     const int warp_id = g_thread_id / (32);
//     const int thread_x = g_thread_id % (32);
//
//     const int start_m = blockIdx.y * BLOCK_DIM_M;
//     const int start_n = blockIdx.x * BLOCK_DIM_N;
//
//     __shared__ int a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//     __shared__ int b_quant_data_sm[BLOCK_DIM_N][BLOCK_DIM_K / sizeof(int) + PAD_SIZE];
//     __shared__ half a_scale_data_sm[BLOCK_DIM_M][QUANT_BLKS_PER_WARP + PAD_SIZE];
//     __shared__ half b_scale_data_sm[BLOCK_DIM_N][QUANT_BLKS_PER_WARP + PAD_SIZE];
//
//     float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};
//     int flip_flag = 0;
//
//     load_tile_double_buffer_swizzle<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//         THREAD_NUM_PER_QUANT_BLOCK,
//         PAD_SIZE, B, G, S>(
//         a_ld, M, thread_x, warp_id, start_m, 0, flip_flag, a, &a_quant_data_sm[0][0], &a_scale_data_sm[0][0]);
//     load_tile_double_buffer_swizzle<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//         THREAD_NUM_PER_QUANT_BLOCK,
//         PAD_SIZE, B, G, S>(
//         b_ld, N, thread_x, warp_id, start_n, 0, flip_flag, b, &b_quant_data_sm[0][0], &b_scale_data_sm[0][0]);
//     __syncthreads();
//
//     for (int k = 0; k <= K; k += ((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K)) {
//         const int k_size = min((BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K, K - k);
//         compute_tile_double_buffer<VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K,
//             QUANT_BLKS_PER_WARP,
//             PAD_SIZE,
//             QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT, THREAD_NUM_PER_QUANT_BLOCK>(
//             k_size, thread_x, warp_id, flip_flag,
//             &a_quant_data_sm[0][0], &a_scale_data_sm[0][0],
//             &b_quant_data_sm[0][0], &b_scale_data_sm[0][0],
//             &c_reg[0][0]);
//
//         const int next_k = k + (BLOCK_DIM_K / QUANT_BLOCK_SIZE) / VEC_DIM_K;
//         if (next_k < K) {
//             //printf("error:  next_k: %d,flip_flag: %d !flip_flag: %d \n", next_k, flip_flag, !flip_flag);
//             load_tile_double_buffer_swizzle<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//                 THREAD_NUM_PER_QUANT_BLOCK,
//                 PAD_SIZE, B, G, S>(
//                 a_ld, M, thread_x, warp_id, start_m, next_k, !flip_flag, a, &a_quant_data_sm[0][0],
//                 &a_scale_data_sm[0][0]);
//             load_tile_double_buffer_swizzle<VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_N, BLOCK_DIM_K, QUANT_BLKS_PER_WARP,
//                 THREAD_NUM_PER_QUANT_BLOCK,
//                 PAD_SIZE, B, G, S>(
//                 b_ld, N, thread_x, warp_id, start_n, next_k, !flip_flag, b, &b_quant_data_sm[0][0],
//                 &b_scale_data_sm[0][0]);
//         }
//         __syncthreads();
//         flip_flag ^= 1;
//     }
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const int dst_m_index = start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
//         if (dst_m_index >= M) {
//             continue;
//         }
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             const int dst_n_index = start_n + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N;
//             if (dst_n_index >= N) {
//                 continue;
//             }
//             if (std::is_same_v<T, half>) {
//                 half old_value = c[dst_m_index * c_ld + dst_n_index];
//                 c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
//             } else if (std::is_same_v<T, float>) {
//                 c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
//             }
//         }
//     }
// }
//
// static int ggml_cuda_dp4a(const int a, const int b, int c) {
//     const int8_t *a8 = (const int8_t *) &a;
//     const int8_t *b8 = (const int8_t *) &b;
//     return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
// }
//
// static void quant_q_8_0_gemm_cpu(int M, int N, int K,
//                                  int a_ld, int b_ld, int c_ld,
//                                  const Q_8 *a,
//                                  const Q_8 *b,
//                                  float *c) {
//     for (int m = 0; m < M; ++m) {
//         for (int n = 0; n < N; ++n) {
//             float sum = 0.0f;
//             for (int k = 0; k < K; ++k) {
//                 const auto &a_block = a[m * a_ld + k];
//                 const auto &b_block = b[n * b_ld + k];
//
//                 const half d_a = __float2half(a_block.d);
//                 const half d_b = __float2half(b_block.d);
//                 const float d_a_f = __half2float(d_a);
//                 const float d_b_f = __half2float(d_b);
//                 const int *a_k_data = reinterpret_cast<const int32_t *>(&a_block.qs[0]);
//
//                 const int *b_k_data = reinterpret_cast<const int32_t *>(&b_block.qs[0]);
//
//                 int dot_q = 0;
//                 for (int i = 0; i < Q_8::BLOCK_SIZE / 4; ++i) {
//                     auto a_value = a_k_data[i];
//                     auto b_value = b_k_data[i];
//                     dot_q = ggml_cuda_dp4a(a_value, b_value, dot_q);
//                 }
//                 sum += d_a_f * d_b_f * static_cast<float>(dot_q);
//             }
//             c[m * c_ld + n] = sum;
//         }
//     }
// }
//
// constexpr int BLOCK_SIZE = 32;
// constexpr int WARP_NUM_PER_BLOCK = 8;
//
// template<const int WARP_SIZE, const int BLOCK_SIZE, const int WARP_NUM_PER_BLOCK>
// __global__ void quant_q_8_0(const float *__restrict__ src,
//                             void *dst, const int64_t M, const int64_t N, const int ld, const int dst_stride_cnt) {
//     const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
//     const int warp_id = g_thread_id / WARP_SIZE;
//     const int lane_id = g_thread_id % WARP_SIZE;
//
//     const int start_block = blockIdx.x * WARP_NUM_PER_BLOCK * WARP_SIZE;
//     const int g_index = start_block + warp_id * BLOCK_SIZE + lane_id;
//
//     const int g_dst_index = blockIdx.x * WARP_NUM_PER_BLOCK + warp_id;
//     auto *dst_ptr = static_cast<Q_8 *>(dst);
//
//     float x = 0.0f;
//     float max_value = 0.0f;
//     if (g_index < N * M) {
//         x = src[g_index];
//         max_value = fabsf(x);
//     }
//
// #pragma unroll
//     for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
//         max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
//     }
//
//     float d = max_value / 127.0f;
//     if (g_dst_index < M * N / BLOCK_SIZE) {
//         if (lane_id == 0) {
//             dst_ptr[g_dst_index].d = __float2half(d);
//         }
//         dst_ptr[g_dst_index].qs[lane_id] =
//                 max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
//     }
// }
//
// template<const int WARP_SIZE, const int BLOCK_SIZE>
// __global__ void quant_q_8_0_2d(const float *__restrict__ src,
//                                void *dst, const int M, const int N, const int ld, const int dst_stride_cnt) {
//     const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
//     const int warp_id = g_thread_id / WARP_SIZE;
//     const int lane_id = g_thread_id % WARP_SIZE;
//
//     const int row = blockIdx.y * blockDim.y + warp_id;
//     const int col = blockIdx.x * blockDim.x + lane_id;
//     const int start_dst_row = row;
//     const int start_dst_col = col / BLOCK_SIZE;
//     auto *dst_ptr = static_cast<Q_8 *>(dst);
//
//     float x = 0.0f;
//     float max_value = 0.0f;
//     if (col < N && row < M) {
//         x = src[row * ld + col];
//         max_value = fabsf(x);
//     }
//
// #pragma unroll
//     for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
//         max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
//     }
//     float d = max_value / 127.0f;
//     const int index = start_dst_row * dst_stride_cnt + start_dst_col;
//     if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
//         if (lane_id == 0) {
//             dst_ptr[index].d = __float2half(d);
//         }
//         dst_ptr[index].qs[lane_id] = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
//     }
// }
//
// #include <cstdint>
//
// static inline uint32_t div_u32_cpu(uint32_t n, uint32_t magic, int shift) {
//     // 计算 (n * magic) 的高32位：等价于 (uint64_t)n * magic >> 32
//     uint64_t product = (uint64_t) n * magic;
//     uint32_t high_part = (uint32_t) (product >> 32);
//     return (high_part + n) >> shift;
// }
//
// static void quantize_func(const float *src, Q_8 *blocks, const int64_t elem_count) {
//     const int nb = static_cast<int>(elem_count / BLOCK_SIZE);
//     for (int i = 0; i < nb; ++i) {
//         const float *x = src + i * BLOCK_SIZE;
//
//         float max_abs = 0.0f;
//         for (int j = 0; j < BLOCK_SIZE; ++j) {
//             max_abs = std::max(max_abs, std::abs(x[j]));
//         }
//         if (max_abs == 0.0f) {
//             blocks[i].d = tff::utils::fp32_to_fp16(0.0f);
//             for (int j = 0; j < BLOCK_SIZE; ++j) {
//                 blocks[i].qs[j] = 0;
//             }
//             continue;
//         }
//         const float scale = max_abs / 127.0f;
//         const float inv_scale = 1.0f / scale;
//
//         blocks[i].d = half(scale);
//
//         for (int j = 0; j < BLOCK_SIZE; ++j) {
//             const float v = x[j] * inv_scale;
//             const int32_t iv = static_cast<int32_t>(std::round(v));
//             blocks[i].qs[j] = static_cast<int8_t>(
//                 std::max(-127, std::min(127, iv))
//             );
//             // if (i == 156 * 512 / BLOCK_SIZE + 10 && j == 19) {
//             //     printf("half d: %f\n", (blocks[i].d));
//             //     printf("d: %f\n", scale);
//             //     printf("x: %f\n", x);
//             //     printf("x / d: %f\n", v);
//             //     printf("roundf(x / d): %f\n", std::round(v));
//             //     printf("static_cast<int32_t>(roundf(x / d)): %d\n", static_cast<int32_t>(std::round(v)));
//             //     printf("static_cast<int8_t>(roundf(x / d)): %d\n", static_cast<int8_t>(static_cast<int32_t>(std::round(v))));
//             // }
//         }
//     }
// }
//
// template<typename T>
// static void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
//     for (auto &element: vector) {
//         element = static_cast<T>(dist(mt));
//     }
// }
//
// static void quant_q_8_0_1d(const int64_t m, const int64_t n, std::vector<float> &src,
//                            std::vector<Q_8> &dst) {
//     dst.resize(m * n / Q_8::BLOCK_SIZE);
//
//     std::vector<Q_8> c_mat_quant_result_cpu;
//     c_mat_quant_result_cpu.resize(m * n / Q_8::BLOCK_SIZE);
// #ifdef _DEBUG
//     quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
// #endif
//     float *src_gpu = nullptr;
//     cudaMalloc(&src_gpu, sizeof(float) * m * n);
//     cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
//     void *dst_gpu = nullptr;
//     cudaMalloc(&dst_gpu, sizeof(Q_8) * m * n / Q_8::BLOCK_SIZE);
//
//     const int WARP_SIZE = 32;
//     dim3 grid((n * m + WARP_NUM_PER_BLOCK * WARP_SIZE - 1) / (WARP_NUM_PER_BLOCK * WARP_SIZE), 1, 1);
//     dim3 block(WARP_NUM_PER_BLOCK * WARP_SIZE, 1, 1);
//     printf("grid:%d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     quant_q_8_0<32, BLOCK_SIZE, WARP_NUM_PER_BLOCK><<<grid, block>>>(src_gpu, dst_gpu, m, n, n, n / BLOCK_SIZE);
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(Q_8),
//                cudaMemcpyKind::cudaMemcpyDeviceToHost);
//     cudaFree(dst_gpu);
//     dst_gpu = nullptr;
//     cudaFree(src_gpu);
//     src_gpu = nullptr;
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//
//     printf("Matrix size: %d x %d\n", m, n);
//     printf("Kernel time: %.4f ms\n", milliseconds);
// #ifdef _DEBUG
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n / BLOCK_SIZE; j++) {
//             for (int k = 0; k < BLOCK_SIZE; k++) {
//                 float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (dst[i * n / BLOCK_SIZE + j].
//                                      qs[k]);
//                 //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
//                 if (delta_qs > 1.0f) {
//                     printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i, j, k, delta_qs);
//                     return;
//                 }
//             }
//             //printf("\n");
//         }
//         //printf("\n");
//     }
// #endif
//     printf("1d success!!\n");
// }
//
// static void quant_q_8_0_2d(const int m, const int n, std::vector<float> &src,
//                            std::vector<Q_8> &dst) {
//     dst.resize(m * n / Q_8::BLOCK_SIZE);
//
//     std::vector<Q_8> c_mat_quant_result_cpu;
//     c_mat_quant_result_cpu.resize(m * n / Q_8::BLOCK_SIZE);
// #ifdef _DEBUG
//     quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
// #endif
//     float *src_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&src_gpu, sizeof(float) * m * n));
//     CudaSafeCall(cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice));
//     void *dst_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&dst_gpu, sizeof(Q_8) * m * n / Q_8::BLOCK_SIZE));
//
//
//     const int WARP_SIZE = 32;
//     dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
//     dim3 block(WARP_SIZE, WARP_NUM_PER_BLOCK, 1);
//     printf("grid:%d, %d, block: %d, %d\n", grid.x, grid.y, block.x, block.y);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//     quant_q_8_0_2d<32, BLOCK_SIZE><<<grid, block>>>(src_gpu, dst_gpu, m, n, n, n / BLOCK_SIZE);
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(Q_8),
//                cudaMemcpyKind::cudaMemcpyDeviceToHost);
//     cudaFree(dst_gpu);
//     dst_gpu = nullptr;
//     cudaFree(src_gpu);
//     src_gpu = nullptr;
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//
//     //printf("Matrix size: %d x %d\n", m, n);
//     //printf("Kernel time: %.4f ms\n", milliseconds);
// #ifdef _DEBUG
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n / BLOCK_SIZE; j++) {
//             for (int k = 0; k < BLOCK_SIZE; k++) {
//                 float delta_qs = (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (dst[i * n / BLOCK_SIZE + j].
//                                      qs[k]);
//                 //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
//                 if (delta_qs > 1.0f) {
//                     printf("m: %d, n: %d, block_index: %d,error qs diff: %d\n", i, j, k, delta_qs);
//                     return;
//                 }
//             }
//             //printf("\n");
//         }
//         //printf("\n");
//     }
// #endif
//     //printf("2d success!!\n");
// }
//
// template<typename T>
// void quant_q_8_0_gemm_nt(const int m, const int n, const int k, const int quant_k,
//                          std::vector<float> &a_mat,
//                          std::vector<float> &b_mat) {
//     std::vector<Q_8> a_quant;
//     quant_q_8_0_2d(m, k, a_mat, a_quant);
//     Q_8 *a_mat_gpu = nullptr;
//     cudaMalloc(&a_mat_gpu, sizeof(Q_8) * m * quant_k);
//     cudaMemcpy(a_mat_gpu, a_quant.data(), sizeof(Q_8) * m * quant_k,
//                cudaMemcpyHostToDevice);
//
//     std::vector<Q_8> b_quant;
//     quant_q_8_0_2d(n, k, b_mat, b_quant);
//     Q_8 *b_mat_gpu = nullptr;
//     cudaMalloc(&b_mat_gpu, sizeof(Q_8) * n * quant_k);
//     cudaMemcpy(b_mat_gpu, b_quant.data(), sizeof(Q_8) * n * quant_k,
//                cudaMemcpyHostToDevice);
//
//     //c
//     T *c_mat_gpu = nullptr;
//     cudaMalloc(&c_mat_gpu, sizeof(T) * m * n);
//     cudaMemset(c_mat_gpu, 0, sizeof(T) * m * n);
//     std::vector<float> c_mat;
//     c_mat.resize(m * n);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//
//     constexpr int QUANT_BLOCK_SIZE = 32;
//     constexpr int VEC_DIM_M = 8;
//     constexpr int VEC_DIM_N = 2;
//     constexpr int VEC_DIM_K = 2;
//     constexpr int BLOCK_M_DIM = 64;
//     constexpr int BLOCK_N_DIM = 64;
//     constexpr int BLOCK_K_DIM = 256;
//     constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
//     constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int);
//     constexpr int PAD_SIZE = 4;
//     constexpr int DOUBLE_BUFFER_PAD_SIZE = 4;
//     constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);
//
//     dim3 grid((n + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (m + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
//     //dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
//     dim3 block(32, QUANT_BLKS_PER_WARP, 1);
//     printf("grid:%d, %d\n", grid.x, grid.y);
//     printf("block:%d, %d\n", block.x, block.y);
//     gemm_quant_q_8_0_nt<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM, BLOCK_K_DIM,
//                 QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, PAD_SIZE, QUANT_BLOCK_SIZE, QUANT_BLOCK_SIZE / sizeof(
//                     int)><<<
//             grid, block>>>(
//                 m, n, quant_k, quant_k, quant_k, n, a_mat_gpu, b_mat_gpu, c_mat_gpu);
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     cudaMemcpy(c_mat.data(), c_mat_gpu, c_mat.size() * sizeof(T),
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
//     printf("手写 quant gemm Performance: %.2f GFLOPS/s\n", gflops_per_sec);
// #ifdef _DEBUG
//     std::vector<float> c_mat_cpu;
//     c_mat_cpu.resize(m * n);
//     quant_q_8_0_gemm_cpu(m, n, quant_k, quant_k,
//                          quant_k, n, a_quant.data(), b_quant.data(), c_mat_cpu.data());
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float delta = c_mat_cpu[i * n + j] - c_mat[i * n + j];
//             if (delta > 1.0f) {
//                 printf("m: %d, n: %d, error: %f, c_mat_cpu[%d]: %f, c_mat[%d]: %f,\n", i, j, delta,
//                        i * n + j, c_mat_cpu[i * n + j], i * n + j, c_mat[i * n + j]);
//                 return;
//             }
//         }
//     }
//     printf("quant gemm sucess!\n");
// #endif
// }
//
// template<typename T>
// void quant_q_8_0_fp32_gemm_nt_double_buffer(const int m, const int n, const int k, const int quant_k,
//                                             std::vector<float> &a_mat,
//                                             std::vector<float> &b_mat) {
//     std::vector<Q_8> a_quant;
//     quant_q_8_0_2d(m, k, a_mat, a_quant);
//     void *a_mat_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&a_mat_gpu, sizeof(Q_8) * m * quant_k));
//     CudaSafeCall(cudaMemcpy(a_mat_gpu, a_quant.data(), sizeof(Q_8) * m * quant_k,
//         cudaMemcpyHostToDevice));
//
//     void *b_mat_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&b_mat_gpu, sizeof(float) * n * k));
//     CudaSafeCall(cudaMemcpy(b_mat_gpu, b_mat.data(), sizeof(float) * n * k,
//         cudaMemcpyHostToDevice));
//
//     //c
//     T *c_mat_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&c_mat_gpu, sizeof(T) * m * n));
//     CudaSafeCall(cudaMemset(c_mat_gpu, 0, sizeof(T) * m * n));
//     std::vector<float> c_mat;
//     c_mat.resize(m * n);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
// #define _big
// #ifdef _big
//     constexpr int QUANT_BLOCK_SIZE = 32;
//     constexpr int VEC_DIM_M = 8;
//     constexpr int VEC_DIM_N = 2;
//     constexpr int VEC_DIM_K = 2;
//     constexpr int BLOCK_M_DIM = 64;
//     constexpr int BLOCK_N_DIM = 64;
//     constexpr int BLOCK_K_DIM = 64;
//     constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
//     constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int);
//     constexpr int PAD_SIZE = 4;
//     constexpr int DOUBLE_BUFFER_PAD_SIZE = 4;
//     constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);
// #else
//     constexpr int QUANT_BLOCK_SIZE = 32;
//     constexpr int VEC_DIM_M = 4;
//     constexpr int VEC_DIM_N = 1;
//     constexpr int VEC_DIM_K = 2;
//     constexpr int BLOCK_M_DIM = 32;
//     constexpr int BLOCK_N_DIM = 32;
//     constexpr int BLOCK_K_DIM = 256;
//     constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
//     constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int);
//     constexpr int PAD_SIZE = 4;
//     constexpr int DOUBLE_BUFFER_PAD_SIZE = 4;
//     constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);
// #endif
//     dim3 grid((n + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (m + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
//     //dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
//     dim3 block(32, QUANT_BLKS_PER_WARP, 1);
//     printf("grid:%d, %d\n", grid.x, grid.y);
//     printf("block:%d, %d\n", block.x, block.y);
//     gemm_quant_q_8_0_fp32_nt_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM, BLOCK_K_DIM,
//         QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, DOUBLE_BUFFER_PAD_SIZE, QUANT_BLOCK_SIZE,
//         QUANT_BLOCK_SIZE / sizeof(int)><<<grid, block>>>(
//         m, n, quant_k, quant_k, quant_k, n, a_mat_gpu, b_mat_gpu, c_mat_gpu);
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     CudaSafeCall(cudaMemcpy(c_mat.data(), c_mat_gpu, c_mat.size() * sizeof(T),
//         cudaMemcpyKind::cudaMemcpyDeviceToHost));
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
//     printf("手写 quant gemm double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);
// #ifdef _DEBUG
//     std::vector<float> c_mat_cpu;
//     c_mat_cpu.resize(m * n);
//     quant_q_8_0_gemm_cpu(m, n, quant_k, quant_k,
//                          quant_k, n, a_quant.data(), b_quant.data(), c_mat_cpu.data());
//     int error_num = 0;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float delta = c_mat_cpu[i * n + j] - c_mat[i * n + j];
//             if (delta > 0.001f) {
//                 error_num++;
//                 printf("m: %d, n: %d, error: %f, c_mat_cpu[%d]: %f, c_mat[%d]: %f,\n", i, j, delta,
//                        i * n + j, c_mat_cpu[i * n + j], i * n + j, c_mat[i * n + j]);
//                 //return;
//             }
//         }
//     }
//     float error_ratio = float(error_num) / static_cast<float>(m * n);
//     if (error_ratio > 0.01) {
//         printf("error\n");
//         return;
//     }
//     printf("quant gemm sucess!\n");
// #endif
// }
//
// template<typename T>
// void quant_q_8_0_gemm_nt_double_buffer(const int m, const int n, const int k, const int quant_k,
//                                        std::vector<float> &a_mat,
//                                        std::vector<float> &b_mat) {
//     std::vector<Q_8> a_quant;
//     quant_q_8_0_2d(m, k, a_mat, a_quant);
//     void *a_mat_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&a_mat_gpu, sizeof(Q_8) * m * quant_k));
//     CudaSafeCall(cudaMemcpy(a_mat_gpu, a_quant.data(), sizeof(Q_8) * m * quant_k,
//         cudaMemcpyHostToDevice));
//
//     std::vector<Q_8> b_quant;
//     quant_q_8_0_2d(n, k, b_mat, b_quant);
//     void *b_mat_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&b_mat_gpu, sizeof(Q_8) * n * quant_k));
//     CudaSafeCall(cudaMemcpy(b_mat_gpu, b_quant.data(), sizeof(Q_8) * n * quant_k,
//         cudaMemcpyHostToDevice));
//
//     //c
//     T *c_mat_gpu = nullptr;
//     CudaSafeCall(cudaMalloc(&c_mat_gpu, sizeof(T) * m * n));
//     CudaSafeCall(cudaMemset(c_mat_gpu, 0, sizeof(T) * m * n));
//     std::vector<float> c_mat;
//     c_mat.resize(m * n);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
// #define _big
// #ifdef _big
//     constexpr int QUANT_BLOCK_SIZE = 32;
//     constexpr int VEC_DIM_M = 8;
//     constexpr int VEC_DIM_N = 2;
//     constexpr int VEC_DIM_K = 2;
//     constexpr int BLOCK_M_DIM = 64;
//     constexpr int BLOCK_N_DIM = 64;
//     constexpr int BLOCK_K_DIM = 256;
//     constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
//     constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int);
//     constexpr int PAD_SIZE = 4;
//     constexpr int DOUBLE_BUFFER_PAD_SIZE = 4;
//     constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);
// #else
//     constexpr int QUANT_BLOCK_SIZE = 32;
//     constexpr int VEC_DIM_M = 4;
//     constexpr int VEC_DIM_N = 1;
//     constexpr int VEC_DIM_K = 2;
//     constexpr int BLOCK_M_DIM = 32;
//     constexpr int BLOCK_N_DIM = 32;
//     constexpr int BLOCK_K_DIM = 256;
//     constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
//     constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int);
//     constexpr int PAD_SIZE = 4;
//     constexpr int DOUBLE_BUFFER_PAD_SIZE = 4;
//     constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);
// #endif
//     dim3 grid((n + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (m + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
//     //dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
//     dim3 block(32, QUANT_BLKS_PER_WARP, 1);
//     printf("grid:%d, %d\n", grid.x, grid.y);
//     printf("block:%d, %d\n", block.x, block.y);
//     gemm_quant_q_8_0_nt_double_buffer<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM, BLOCK_K_DIM,
//         QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, DOUBLE_BUFFER_PAD_SIZE, QUANT_BLOCK_SIZE,
//         QUANT_BLOCK_SIZE / sizeof(int)><<<grid, block>>>(
//         m, n, quant_k, quant_k, quant_k, n, a_mat_gpu, b_mat_gpu, c_mat_gpu);
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     CudaSafeCall(cudaMemcpy(c_mat.data(), c_mat_gpu, c_mat.size() * sizeof(T),
//         cudaMemcpyKind::cudaMemcpyDeviceToHost));
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
//     printf("手写 quant gemm double buffer Performance: %.2f GFLOPS/s\n", gflops_per_sec);
// #ifdef _DEBUG
//     std::vector<float> c_mat_cpu;
//     c_mat_cpu.resize(m * n);
//     quant_q_8_0_gemm_cpu(m, n, quant_k, quant_k,
//                          quant_k, n, a_quant.data(), b_quant.data(), c_mat_cpu.data());
//     int error_num = 0;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float delta = c_mat_cpu[i * n + j] - c_mat[i * n + j];
//             if (delta > 0.001f) {
//                 error_num++;
//                 printf("m: %d, n: %d, error: %f, c_mat_cpu[%d]: %f, c_mat[%d]: %f,\n", i, j, delta,
//                        i * n + j, c_mat_cpu[i * n + j], i * n + j, c_mat[i * n + j]);
//                 //return;
//             }
//         }
//     }
//     float error_ratio = float(error_num) / static_cast<float>(m * n);
//     if (error_ratio > 0.01) {
//         printf("error\n");
//         return;
//     }
//     printf("quant gemm sucess!\n");
// #endif
// }
//
// template<typename T>
// void quant_q_8_0_gemm_nt_double_buffer_swizzle(const int m, const int n, const int k, const int quant_k,
//                                                std::vector<float> &a_mat,
//                                                std::vector<float> &b_mat) {
//     std::vector<Q_8> a_quant;
//     quant_q_8_0_2d(m, k, a_mat, a_quant);
//     Q_8 *a_mat_gpu = nullptr;
//     cudaMalloc(&a_mat_gpu, sizeof(Q_8) * m * quant_k);
//     cudaMemcpy(a_mat_gpu, a_quant.data(), sizeof(Q_8) * m * quant_k,
//                cudaMemcpyHostToDevice);
//
//     std::vector<Q_8> b_quant;
//     quant_q_8_0_2d(n, k, b_mat, b_quant);
//     Q_8 *b_mat_gpu = nullptr;
//     cudaMalloc(&b_mat_gpu, sizeof(Q_8) * n * quant_k);
//     cudaMemcpy(b_mat_gpu, b_quant.data(), sizeof(Q_8) * n * quant_k,
//                cudaMemcpyHostToDevice);
//
//     //c
//     T *c_mat_gpu = nullptr;
//     cudaMalloc(&c_mat_gpu, sizeof(T) * m * n);
//     cudaMemset(c_mat_gpu, 0, sizeof(T) * m * n);
//     std::vector<float> c_mat;
//     c_mat.resize(m * n);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//
//     constexpr int QUANT_BLOCK_SIZE = 32;
//     constexpr int VEC_DIM_M = 8;
//     constexpr int VEC_DIM_N = 2;
//     constexpr int VEC_DIM_K = 2;
//     constexpr int BLOCK_M_DIM = 64;
//     constexpr int BLOCK_N_DIM = 64;
//     constexpr int BLOCK_K_DIM = 256;
//     constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
//     constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int);
//     constexpr int PAD_SIZE = 4;
//     constexpr int DOUBLE_BUFFER_PAD_SIZE = 4;
//     constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);
//
//     dim3 grid((n + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (m + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
//     //dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
//     dim3 block(32, QUANT_BLKS_PER_WARP, 1);
//     printf("grid:%d, %d\n", grid.x, grid.y);
//     printf("block:%d, %d\n", block.x, block.y);
//
//     constexpr int B = compute_bit(VEC_DIM_M);
//     constexpr int M = compute_bit(sizeof(int4) / sizeof(int));
//     constexpr int S = compute_bit(BLOCK_K_DIM / VEC_DIM_K / sizeof(int)) - M;
//
//     gemm_quant_q_8_0_nt_double_buffer_swizzle<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM, BLOCK_K_DIM,
//         QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, DOUBLE_BUFFER_PAD_SIZE, QUANT_BLOCK_SIZE,
//         QUANT_BLOCK_SIZE / sizeof(int), B, M, S><<<grid, block>>>(
//         m, n, quant_k, quant_k, quant_k, n, a_mat_gpu, b_mat_gpu, c_mat_gpu);
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     cudaMemcpy(c_mat.data(), c_mat_gpu, c_mat.size() * sizeof(T),
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
//     printf("手写 quant gemm double buffer swizzle Performance: %.2f GFLOPS/s\n", gflops_per_sec);
// #ifdef _DEBUG
//     std::vector<float> c_mat_cpu;
//     c_mat_cpu.resize(m * n);
//     quant_q_8_0_gemm_cpu(m, n, quant_k, quant_k,
//                          quant_k, n, a_quant.data(), b_quant.data(), c_mat_cpu.data());
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float delta = c_mat_cpu[i * n + j] - c_mat[i * n + j];
//             if (delta > 1.0f) {
//                 printf("m: %d, n: %d, error: %f, c_mat_cpu[%d]: %f, c_mat[%d]: %f,\n", i, j, delta,
//                        i * n + j, c_mat_cpu[i * n + j], i * n + j, c_mat[i * n + j]);
//                 return;
//             }
//         }
//     }
//     printf("quant gemm sucess!\n");
// #endif
// }
//
// template<typename T>
// void quant_q_8_0_gemm_nt_double_buffer_vec(const int m, const int n, const int k, const int quant_k,
//                                            std::vector<float> &a_mat,
//                                            std::vector<float> &b_mat) {
//     std::vector<Q_8> a_quant;
//     quant_q_8_0_2d(m, k, a_mat, a_quant);
//     Q_8 *a_mat_gpu = nullptr;
//     cudaMalloc(&a_mat_gpu, sizeof(Q_8) * m * quant_k);
//     cudaMemcpy(a_mat_gpu, a_quant.data(), sizeof(Q_8) * m * quant_k,
//                cudaMemcpyHostToDevice);
//
//     std::vector<Q_8> b_quant;
//     quant_q_8_0_2d(n, k, b_mat, b_quant);
//     Q_8 *b_mat_gpu = nullptr;
//     cudaMalloc(&b_mat_gpu, sizeof(Q_8) * n * quant_k);
//     cudaMemcpy(b_mat_gpu, b_quant.data(), sizeof(Q_8) * n * quant_k,
//                cudaMemcpyHostToDevice);
//
//     //c
//     T *c_mat_gpu = nullptr;
//     cudaMalloc(&c_mat_gpu, sizeof(T) * m * n);
//     cudaMemset(c_mat_gpu, 0, sizeof(T) * m * n);
//     std::vector<float> c_mat;
//     c_mat.resize(m * n);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//
//     constexpr int QUANT_BLOCK_SIZE = 32;
//     constexpr int VEC_DIM_M = 4;
//     constexpr int VEC_DIM_N = 1;
//     constexpr int VEC_DIM_K = 2;
//     constexpr int BLOCK_M_DIM = 32;
//     constexpr int BLOCK_N_DIM = 32;
//     constexpr int BLOCK_K_DIM = 512;
//     constexpr int QUANT_BLKS_PER_WARP = BLOCK_K_DIM / QUANT_BLOCK_SIZE;
//     constexpr int THREAD_NUM_PER_QUANT_BLOCK = QUANT_BLOCK_SIZE / sizeof(int4);
//     constexpr int PAD_SIZE = 0;
//     constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int4);
//
//     dim3 grid((n + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (m + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
//     //dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
//     dim3 block(32, 8, 1);
//     printf("grid:%d, %d\n", grid.x, grid.y);
//     printf("block:%d, %d\n", block.x, block.y);
//
//     constexpr int B = compute_bit(VEC_DIM_M);
//     constexpr int M = compute_bit(sizeof(int4) / sizeof(int));
//     constexpr int S = compute_bit(BLOCK_K_DIM / VEC_DIM_K / sizeof(int)) - M;
//
//     const int tmp = BLOCK_K_DIM / sizeof(int4);
//     gemm_quant_q_8_0_nt_double_buffer_vec<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM, BLOCK_K_DIM,
//         QUANT_BLKS_PER_WARP, THREAD_NUM_PER_QUANT_BLOCK, PAD_SIZE, QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT><<<grid, block>>>(
//         m, n, quant_k, quant_k, quant_k, n, a_mat_gpu, b_mat_gpu, c_mat_gpu);
//
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     cudaMemcpy(c_mat.data(), c_mat_gpu, c_mat.size() * sizeof(T),
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
//     printf("手写 quant gemm double buffer vec Performance: %.2f GFLOPS/s\n", gflops_per_sec);
// #ifdef _DEBUG
//     std::vector<float> c_mat_cpu;
//     c_mat_cpu.resize(m * n);
//     quant_q_8_0_gemm_cpu(m, n, quant_k, quant_k,
//                          quant_k, n, a_quant.data(), b_quant.data(), c_mat_cpu.data());
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             float delta = c_mat_cpu[i * n + j] - c_mat[i * n + j];
//             if (delta > 1.0f) {
//                 printf("m: %d, n: %d, error: %f, c_mat_cpu[%d]: %f, c_mat[%d]: %f,\n", i, j, delta,
//                        i * n + j, c_mat_cpu[i * n + j], i * n + j, c_mat[i * n + j]);
//                 return;
//             }
//         }
//     }
//     printf("quant gemm sucess!\n");
// #endif
// }
//
// int main325435(int argc, char *argv[]) {
//     cudaDeviceProp device_prop{};
//     cudaGetDeviceProperties(&device_prop, 0);
//     std::mt19937 mt(42);
//     std::uniform_real_distribution<double> dist(-4, 4);
// #ifdef _DEBUG
//     int dim = 513;
//     int m = dim;
//     int n = dim;
//     int k = 512;
//     int quant_k = k / Q_8::BLOCK_SIZE;
//
//     std::vector<float> a_mat;
//     a_mat.resize(m * k);
//     PopulateVector<float>(a_mat, mt, dist);
//
//
//     std::vector<float> b_mat;
//     b_mat.resize(n * k);
//     PopulateVector<float>(b_mat, mt, dist);
//
//     //quant_q_8_0_gemm_nt<T>(m, n, k, quant_k, a_mat, b_mat);
//     quant_q_8_0_gemm_nt_double_buffer<T>(m, n, k, quant_k, a_mat, b_mat);
//     //quant_q_8_0_gemm_nt_double_buffer_swizzle<T>(m, n, k, quant_k, a_mat, b_mat);
//     //quant_q_8_0_gemm_nt_double_buffer_vec<T>(m, n, k, quant_k, a_mat, b_mat);
// #else
//     int dim = 256;
//     int m = dim;
//     int n = dim;
//     int k = dim;
//     int quant_k = k / Q_8::BLOCK_SIZE;
//
//     std::vector<float> a_mat;
//     a_mat.resize(m * k);
//     PopulateVector<float>(a_mat, mt, dist);
//
//
//     std::vector<float> b_mat;
//     b_mat.resize(n * k);
//     PopulateVector<float>(b_mat, mt, dist);
//
//     quant_q_8_0_gemm_nt<T>(m, n, k, quant_k, a_mat, b_mat);
//     quant_q_8_0_gemm_nt_double_buffer<T>(m, n, k, quant_k, a_mat, b_mat);
//     //quant_q_8_0_gemm_nt_double_buffer_swizzle<T>(m, n, k, quant_k, a_mat, b_mat);
// #endif
//     return 0;
// }
