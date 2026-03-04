// //
// // Created by nkk on 2026/1/1.
// //
// //
// // Created by nkk on 2025/12/22.
// //
//
// #include <vector>
// #include <random>
// #include "cublas_v2.h"
// #include "mma.h"
// #include <cstdint>
// #include <cstring>
// #include "core/quant/BaseDefine.h"
// #include "device/cuda/cudaInc.h"
// #include "include/kernel_util.h"
// using namespace tff::core::quant;
// using T = half;
// constexpr int QUANT_BLOCK_SIZE = 32;
// constexpr int VEC_DIM_M = 16;
// constexpr int VEC_DIM_N = 4;
// constexpr int VEC_DIM_K = 1;
// constexpr int BLOCK_M_DIM = 128;
// constexpr int BLOCK_N_DIM = 128;
// constexpr int BLOCK_K_DIM = 32;
// constexpr int QUANT_BLOCK_PER_THREAD_BLOCK = BLOCK_K_DIM / VEC_DIM_K;
// constexpr int PAD_SIZE = 4;
// constexpr int BLOCK_PAD_SIZE = BLOCK_K_DIM + PAD_SIZE;
// constexpr int BLOCK_SCALE_PAD_SIZE = QUANT_BLOCK_SIZE / (BLOCK_K_DIM / (VEC_DIM_K * sizeof(int))) + PAD_SIZE;
// constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);
//
// template<const int QUANT_BLOCK_SIZE>
// static __device__ void load_tile_a_quant_vec(const int ld, const int dim,
//                                   const int thread_x, const int warp_id,
//                                   const int start_block,
//                                   const int k,
//                                   const int *__restrict__ global_mem,
//                                   int *sm) {
//     const int* __restrict__ global_int = reinterpret_cast<const int*>(global_mem);
//     const int vec_ld = ld / 4;
//
//     for (int j = 0; j < VEC_DIM_M; ++j) {
//         int dim0 = start_block + warp_id + j * BLOCK_M_DIM / VEC_DIM_M;
//         for (int i = 0; i < VEC_DIM_K; ++i) {
//             int dim1 = (k + thread_x + i * BLOCK_K_DIM / VEC_DIM_K);
//             int val = 0;
//             if (dim1 < vec_ld  && dim0 < dim) {
//                 val = global_int[dim0 * vec_ld + dim1];
//             }
//             sm[(warp_id + j * BLOCK_M_DIM / VEC_DIM_M) * BLOCK_PAD_SIZE + thread_x + i * BLOCK_K_DIM / VEC_DIM_K + 0] =
//                     val;
//         }
//     }
// }
//
// template<const int QUANT_BLOCK_SIZE>
// static __device__ void load_tile_a_scale(const int ld, const int dim,
//                                   const int thread_x, const int warp_id,
//                                   const int start_block,
//                                   const int k,
//                                   const half *__restrict__ global_mem,
//                                   half *sm) {
//     for (int j = 0; j < VEC_DIM_M; ++j) {
//         int dim0 = start_block + warp_id + j * BLOCK_M_DIM / VEC_DIM_M;
//         for (int i = 0; i < (VEC_DIM_K); ++i) {
//             int dim1 = k + thread_x + i * BLOCK_K_DIM / VEC_DIM_K;
//             half val = 0;
//             if (dim1 < ld && dim0 < dim) {
//                 val = global_mem[dim0 * ld + dim1];
//             }
//             sm[(warp_id + j * BLOCK_M_DIM / VEC_DIM_M) * BLOCK_SCALE_PAD_SIZE + thread_x + i * BLOCK_K_DIM / VEC_DIM_K]
//                     =
//                     val;
//         }
//     }
// }
//
// static __device__ int vec_dot_product(int a, int b, int c_sum) {
//     return __dp4a(a, b, c_sum);
// }
//
// template<const int QUANT_BLOCK_SIZE>
// static __device__ void compute_tile(const int thread_x, const int thread_y,
//                              half *a_scale_sm,
//                              half *b_scale_sm,
//                              int *a_quant_sm,
//                              int *b_quant_sm,
//                              float *c_reg) {
//
//
//
// //#pragma unroll
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
// //#pragma unroll
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             float sum = 0;
// #pragma unroll
//             for (int kk = 0; kk < BLOCK_K_DIM / VEC_DIM_K; kk++) {
//
//                 float a_scale[VEC_DIM_M];
//                 float b_scale[VEC_DIM_N];
//                 a_scale[mm] = __half2float(
//                         a_scale_sm[(thread_y + mm * BLOCK_M_DIM / VEC_DIM_M) * BLOCK_SCALE_PAD_SIZE + kk]);
//                 b_scale[nn] = __half2float(
//                         b_scale_sm[(thread_x + nn * BLOCK_N_DIM / VEC_DIM_N) * BLOCK_SCALE_PAD_SIZE + kk]);
//                 int block_sum = 0;
// #pragma unroll
//                 for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q++) {
//                     const int kk_index = kk * VEC_DOT_PRODUCT + kk_q;
//                     int a_reg[VEC_DIM_M] = {0};
//                     int b_reg[VEC_DIM_N] = {0};
//                     a_reg[mm] = a_quant_sm[(thread_y + mm * BLOCK_M_DIM / VEC_DIM_M) * BLOCK_PAD_SIZE + kk_index];
//                     b_reg[nn] = b_quant_sm[(thread_x + nn * BLOCK_N_DIM / VEC_DIM_N) * BLOCK_PAD_SIZE + kk_index];
//
//                     block_sum = vec_dot_product(a_reg[mm], b_reg[nn], block_sum);
//                     // if (blockIdx.y == 0 && blockIdx.x == 25 && thread_y == 12 && mm == 1 && nn == 0 && thread_x == 10) {
//                     //     printf("thread_x: %d, thread_y: %d, a_reg[%d]: %d, b_reg[%d]: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf \n",thread_x, thread_y, mm, a_reg[mm], nn, b_reg[nn],mm, a_scale[mm], nn,b_scale[nn]);
//                     // }
//                 }
//                 // if (blockIdx.y == 0 && blockIdx.x == 25 && thread_y == 12 && mm == 1 && nn == 0 && thread_x == 10) {
//                 //     printf("kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf \n",kk, thread_x, thread_y, block_sum,mm, a_scale[mm], nn,b_scale[nn]);
//                 // }
//                 sum += (block_sum) * (a_scale[mm]) * (b_scale[nn]);
//                 // if (blockIdx.y == 0 && blockIdx.x == 25 && thread_y == 12 && mm == 1 && nn == 0 && thread_x == 10) {
//                 //     printf("kk: %d, thread_x: %d, thread_y: %d, block_sum: %d ,a_scale[%d]: %lf, b_scale[%d]: %lf ,sum:%lf \n",kk, thread_x, thread_y, block_sum,mm, a_scale[mm], nn,b_scale[nn], sum);
//                 // }
//             }
//
//             c_reg[mm * VEC_DIM_N + nn] += sum;
//         }
//     }
// }
//
// template<typename T, const int QUANT_BLOCK_SIZE>
// __global__ void gemm_quant_q_8_0_nt_reshape(
//     int M, int N, int K,
//     int a_ld, int b_ld, int c_ld,
//     const half *__restrict__ a_scale,
//     const half *__restrict__ b_scale,
//     const int32_t *__restrict__ a_quant_data,
//     const int32_t *__restrict__ b_quant_data,
//     T *__restrict__ c) {
//     const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;
//
//     const int thread_y = g_thread_id / (BLOCK_N_DIM / VEC_DIM_N);
//     const int thread_x = g_thread_id % (BLOCK_N_DIM / VEC_DIM_N);
//
//     const int start_m = blockIdx.y * BLOCK_M_DIM;
//     const int start_n = blockIdx.x * BLOCK_N_DIM;
//
//
//     __shared__ int a_quant_data_sm[BLOCK_M_DIM][BLOCK_PAD_SIZE];
//     __shared__ int b_quant_data_sm[BLOCK_N_DIM][BLOCK_PAD_SIZE];
//     __shared__ half a_scale_data_sm[BLOCK_M_DIM][BLOCK_SCALE_PAD_SIZE];
//     __shared__ half b_scale_data_sm[BLOCK_N_DIM][BLOCK_SCALE_PAD_SIZE];
//     float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};
//
//     for (size_t k = 0; k < K; k += BLOCK_K_DIM / VEC_DIM_K) {
//         load_tile_a_quant_vec<QUANT_BLOCK_SIZE>(a_ld, M, thread_x, thread_y, start_m, k, a_quant_data,
//                                             &a_quant_data_sm[0][0]);
//         load_tile_a_quant_vec<QUANT_BLOCK_SIZE>(b_ld, N, thread_x, thread_y, start_n, k, b_quant_data,
//                                             &b_quant_data_sm[0][0]);
//         load_tile_a_scale<QUANT_BLOCK_SIZE>(a_ld, M, thread_x, thread_y, start_m, k, a_scale, &a_scale_data_sm[0][0]);
//         load_tile_a_scale<QUANT_BLOCK_SIZE>(b_ld, N, thread_x, thread_y, start_n, k, b_scale, &b_scale_data_sm[0][0]);
//         __syncthreads();
//
//         compute_tile<QUANT_BLOCK_SIZE>(thread_x, thread_y, &a_scale_data_sm[0][0], &b_scale_data_sm[0][0],
//                                        &a_quant_data_sm[0][0], &b_quant_data_sm[0][0], &c_reg[0][0]);
//         __syncthreads();
//     }
//
//     for (int mm = 0; mm < VEC_DIM_M; mm++) {
//         const int dst_m_index = start_m + thread_y + mm * BLOCK_M_DIM / VEC_DIM_M;
//         if (dst_m_index >= M) {
//             continue;
//         }
//         for (int nn = 0; nn < VEC_DIM_N; nn++) {
//             const int dst_n_index = start_n + thread_x + nn * BLOCK_N_DIM / VEC_DIM_N;
//             if (dst_n_index >= N) {
//                 continue;
//             }
//             if (std::is_same_v<T, half>) {
//                 half old_value = c[dst_m_index * c_ld + dst_n_index];
//                 c[dst_m_index * c_ld + dst_n_index] = old_value + __float2half(c_reg[mm][nn]);
//             }else if (std::is_same_v<T, float>) {
//                 c[dst_m_index * c_ld + dst_n_index] += c_reg[mm][nn];
//             }
//
//         }
//     }
// }
//
// static int ggml_cuda_dp4a(const int a, const int b, int c) {
//     const int8_t *a8 = (const int8_t *) &a;
//     const int8_t *b8 = (const int8_t *) &b;
//     return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
// }
// template<typename T>
// static void quant_q_8_0_gemm_cpu(int M, int N, int K,
//                                  int a_ld, int b_ld, int c_ld,
//                                  const void *a,
//                                  const void *b,
//                                  T *c) {
//     half *a_scale = (half*)a;
//     half *b_scale = (half*)b;
//     int *a_quant = (int *) (a + M * K * sizeof(half));
//     int *b_quant = (int *) (b + N * K * sizeof(half));
//
//     for (int m = 0; m < M; ++m) {
//         for (int n = 0; n < N; ++n) {
//             float sum = 0.0f;
//             for (int k = 0; k < K; ++k) {
//                 const auto a_block = &a_quant[m * a_ld * 8 + k * 8];
//                 const auto b_block = &b_quant[n * b_ld * 8 + k * 8];
//
//                 const float d_a = __half2float(a_scale[m * a_ld + k]);
//                 const float d_b = __half2float(b_scale[n * b_ld + k]);
//
//                 float dot_q = 0;
//                 for (int i = 0; i < tff::core::quant::Q_8_0::BLOCK_SIZE / 4; ++i) {
//                     auto a_value = a_block[i];
//                     auto b_value = b_block[i];
//                     dot_q = ggml_cuda_dp4a(a_value, b_value, dot_q);
//                 }
//                 sum += d_a * d_b * (dot_q);
//             }
//             if (std::is_same_v<T, half>) {
//                 half tmp = __float2half(27223.996094f);
//                 c[m * c_ld + n] = __float2half(sum);
//
//             }else if (std::is_same_v<T, float>) {
//                 c[m * c_ld + n] = sum;
//             }
//
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
//     auto *dst_ptr = static_cast<tff::core::quant::Q_8_0 *>(dst);
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
//     auto *dst_ptr = static_cast<tff::core::quant::Q_8_0 *>(dst);
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
// template<const int WARP_SIZE, const int BLOCK_SIZE>
// __global__ void quant_q_8_0_2d_reshape(const float *__restrict__ src,
//                                        void *scale_ptr,
//                                        void *quant_data_ptr, const int M, const int N, const int ld,
//                                        const int dst_stride_cnt) {
//     const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
//     const int warp_id = g_thread_id / WARP_SIZE;
//     const int lane_id = g_thread_id % WARP_SIZE;
//
//     const int row = blockIdx.y * blockDim.y + warp_id;
//     const int col = blockIdx.x * blockDim.x + lane_id;
//     const int start_dst_row = row;
//     const int start_dst_col = col / BLOCK_SIZE;
//     auto *scale = static_cast<half *>(scale_ptr);
//     auto *quant_data = static_cast<int8_t *>(quant_data_ptr);
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
//     int8_t *current_quant_block = quant_data + index * BLOCK_SIZE;
//     if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
//         if (lane_id == 0) {
//             scale[index] = __float2half(d);
//         }
//         current_quant_block[lane_id] = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
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
// static void quantize_func(const float *src, Q_8_0 *blocks, const int64_t elem_count) {
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
//             const int32_t iv = static_cast<int32_t>(std::roundf(v));
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
//                     std::vector<tff::core::quant::Q_8_0> &dst) {
//     dst.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
//
//     std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
//     c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
// #ifdef _DEBUG
//     quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
// #endif
//     float *src_gpu = nullptr;
//     cudaMalloc(&src_gpu, sizeof(float) * m * n);
//     cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
//     void *dst_gpu = nullptr;
//     cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
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
//     cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(tff::core::quant::Q_8_0),
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
// static void quant_q_8_0_2d(const int m, const int n, std::vector<float> &src, std::vector<tff::core::quant::Q_8_0> &dst) {
//     dst.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
//
//     std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
//     c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
// #ifdef _DEBUG
//     quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
// #endif
//     float *src_gpu = nullptr;
//     cudaMalloc(&src_gpu, sizeof(float) * m * n);
//     cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
//     void *dst_gpu = nullptr;
//     cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
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
//     cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(tff::core::quant::Q_8_0),
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
//     printf("2d success!!\n");
// }
//
// static void quant_q_8_0_2d_reshape(const int m, const int n, std::vector<float> &src,
//                             std::vector<tff::core::quant::Q_8_0> &dst) {
//     dst.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
//
//     std::vector<tff::core::quant::Q_8_0> c_mat_quant_result_cpu;
//     c_mat_quant_result_cpu.resize(m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
// #ifdef _DEBUG
//     quantize_func(src.data(), c_mat_quant_result_cpu.data(), m * n);
// #endif
//     float *src_gpu = nullptr;
//     cudaMalloc(&src_gpu, sizeof(float) * m * n);
//     cudaMemcpy(src_gpu, src.data(), sizeof(float) * m * n, cudaMemcpyHostToDevice);
//     void *dst_gpu = nullptr;
//     cudaMalloc(&dst_gpu, sizeof(tff::core::quant::Q_8_0) * m * n / tff::core::quant::Q_8_0::BLOCK_SIZE);
//     void *scale_ptr = dst_gpu;
//     void *quant_data_ptr = dst_gpu + m * n / tff::core::quant::Q_8_0::BLOCK_SIZE * sizeof(half);
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
//     quant_q_8_0_2d_reshape<32, BLOCK_SIZE><<<grid, block>>
//             >(src_gpu, scale_ptr, quant_data_ptr, m, n, n, n / BLOCK_SIZE);
//     cudaEventRecord(stop);
//     cudaDeviceSynchronize();
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     cudaMemcpy(dst.data(), dst_gpu, dst.size() * sizeof(tff::core::quant::Q_8_0),
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
//     auto scaled_gpu = reinterpret_cast<half *>(dst.data());
//     auto quant_data_gpu = static_cast<int8_t *>(reinterpret_cast<void *>(dst.data()) + m * n /
//                                                 tff::core::quant::Q_8_0::BLOCK_SIZE * sizeof(half));
//     float l2error = 0.0f;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n / BLOCK_SIZE; j++) {
//             for (int k = 0; k < BLOCK_SIZE; k++) {
//                 const float delta_qs = (int8_t) (c_mat_quant_result_cpu[i * n / BLOCK_SIZE + j].qs[k]) - (int8_t) (
//                                            quant_data_gpu[i * n + j * BLOCK_SIZE + k]);
//                 //float delta_d = c_mat_quant_result_cpu[i * n + j].d - c_mat_quant_result[i * n + j].d;
//                 if (delta_qs > 1.0f) {
//                     printf("m: %d, n: %d, block_index: %d,error qs diff: %lf\n", i, j, k, delta_qs);
//                     return;
//                 }
//             }
//             //printf("\n");
//         }
//         //printf("\n");
//     }
// #endif
//     printf("2d success!!\n");
// }
// extern "C" static int quant_q_8_0_gemm_main(int argc, char *argv[]);
// int main4567(int argc, char *argv[]) {
//     //quant_q_8_0_gemm_main(argc, argv);
//
//     cudaDeviceProp device_prop{};
//     cudaGetDeviceProperties(&device_prop, 0);
//     std::mt19937 mt(42);
//     std::uniform_real_distribution<double> dist(-127, 127);
//     int dim = 2048;
//     int m = dim;
//     int n = dim;
//     int k = dim;
//     int quant_k = k / tff::core::quant::Q_8_0::BLOCK_SIZE;
//
//     std::vector<float> a_mat;
//     a_mat.resize(m * k);
//     PopulateVector<float>(a_mat, mt, dist);
//     std::vector<tff::core::quant::Q_8_0> a_quant;
//     quant_q_8_0_2d_reshape(m, k, a_mat, a_quant);
//     void *a_mat_gpu = nullptr;
//     cudaMalloc(&a_mat_gpu, sizeof(tff::core::quant::Q_8_0) * m * quant_k);
//     cudaMemcpy(a_mat_gpu, a_quant.data(), sizeof(tff::core::quant::Q_8_0) * m * quant_k,
//                cudaMemcpyHostToDevice);
//     auto a_scaled_gpu = reinterpret_cast<half *>(a_mat_gpu);
//     auto a_quant_gpu = reinterpret_cast<int8_t *>(a_mat_gpu + m * quant_k * sizeof(half));
//
//     std::vector<float> b_mat;
//     b_mat.resize(n * k);
//     PopulateVector<float>(b_mat, mt, dist);
//     std::vector<tff::core::quant::Q_8_0> b_quant;
//     quant_q_8_0_2d_reshape(n, k, b_mat, b_quant);
//     void *b_mat_gpu = nullptr;
//     cudaMalloc(&b_mat_gpu, sizeof(tff::core::quant::Q_8_0) * n * quant_k);
//     cudaMemcpy(b_mat_gpu, b_quant.data(), sizeof(tff::core::quant::Q_8_0) * n * quant_k,
//                cudaMemcpyHostToDevice);
//     auto b_scaled_gpu = reinterpret_cast<half *>(b_mat_gpu);
//     auto b_quant_gpu = reinterpret_cast<int8_t *>(b_mat_gpu + n * quant_k * sizeof(half));
//
//     //c
//     T *c_mat_gpu = nullptr;
//     cudaMalloc(&c_mat_gpu, sizeof(T) * m * n);
//     cudaMemset(c_mat_gpu, 0, sizeof(T) * m * n);
//     std::vector<T> c_mat;
//     c_mat.resize(m * n);
//
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start);
//
//     dim3 grid((n + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (m + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
//     //dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
//     dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
//     printf("grid:%d, %d\n", grid.x, grid.y);
//     printf("block:%d, %d\n", block.x, block.y);
//     gemm_quant_q_8_0_nt_reshape<T, 32><<<grid, block>>>(m, n, quant_k, quant_k, quant_k, n, a_scaled_gpu, b_scaled_gpu,
//                                                      reinterpret_cast<int *>(a_quant_gpu),
//                                                      reinterpret_cast<int *>(b_quant_gpu), c_mat_gpu);
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
//     std::vector<T> c_mat_cpu;
//     c_mat_cpu.resize(m * n, 0);
//     quant_q_8_0_gemm_cpu<T>(m, n, quant_k, quant_k,
//                          quant_k, n, a_quant.data(), b_quant.data(), c_mat_cpu.data());
//
//     float l2error = 0.0f;
//     for (int i = 0; i < m; i++) {
//         for (int j = 0; j < n; j++) {
//             if (std::is_same_v<T, half>) {
//                 float tmp = __half2float(c_mat_cpu[i * n + j]) - __half2float(c_mat[i * n + j]);
//                 l2error += tmp * tmp;
//             }else if (std::is_same_v<T, float>) {
//                 float delta = (c_mat_cpu[i * n + j]) - (c_mat[i * n + j]);
//                 if (delta > 0.01f) {
//                     printf("m: %d, n: %d, error: %f, c_mat_cpu[%d]: %f, c_mat[%d]: %f,\n", i, j, delta,
//                            i * n + j, (c_mat_cpu[i * n + j]), i * n + j, (c_mat[i * n + j]));
//                     return 0;
//                 }
//             }
//
//         }
//     }
//     l2error /= m * n;
//     if (std::is_same_v<T, half>) {
//         if (l2error > 0.05f) {
//             printf("error: %lf", l2error);
//             return 0;
//         }
//     }
//     printf("quant gemm sucess!\n");
// #endif
//     return 0;
// }
