//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
namespace tff::kernel {
    template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int ELEMENTS_PER_LOAD, const int PAD_SIZE>
    __device__ void load_tile_vec_row_major_n(const int ld, const int dim,
                                              const int thread_x, const int warp_id,
                                              const int start_m,
                                              const int k,
                                              const T *__restrict__ global_mem,
                                              T *sm) {
#pragma unroll
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            const int dim0_base = start_m + warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
            for (int kk = 0; kk < VEC_DIM_K / ELEMENTS_PER_LOAD; ++kk) {
                const int dim1 = k + thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD;
                T val[ELEMENTS_PER_LOAD] = {0};
                if (dim0_base < dim) {
                    const int actual_load = min(ELEMENTS_PER_LOAD, ld - dim1);
                    if (actual_load > 0) {
                        load_vec<T>(&global_mem[dim0_base * ld + dim1], val, actual_load);
                    }
                }
                int sm_col = warp_id + j * (BLOCK_DIM_LD / VEC_DIM_LD);
                int sm_row_base = thread_x * ELEMENTS_PER_LOAD + kk * (BLOCK_DIM_K / VEC_DIM_K) * ELEMENTS_PER_LOAD;
#pragma unroll
                for (int i = 0; i < ELEMENTS_PER_LOAD; ++i) {
                    sm[sm_col + (sm_row_base + i) * (BLOCK_DIM_LD + PAD_SIZE)] = val[i];
                }
            }
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int PAD_SIZE>
    __device__ void compute_tile_attention_gemm(const float scale, const int k_size, const int thread_x,
                                                const int warp_id,
                                                T *a_sm, T *b_sm,
                                                float *c_reg) {
#pragma unroll
        for (int kk = 0; kk < k_size; kk++) {
#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                float a_reg = 0.0f;
                if constexpr (std::is_same_v<T, half>) {
                    a_reg = __half2float(
                        a_sm[(kk) * (BLOCK_DIM_M + PAD_SIZE) + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M]);
                } else if constexpr (std::is_same_v<T, float>) {
                    a_reg = (a_sm[(kk) * (BLOCK_DIM_M + PAD_SIZE) + thread_x + mm *
                                  BLOCK_DIM_M /
                                  VEC_DIM_M]);
                }
#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N; nn++) {
                    float b_reg = 0.0f;
                    if constexpr (std::is_same_v<T, half>) {
                        b_reg = __half2float(
                            b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                    } else if constexpr (std::is_same_v<T, float>) {
                        b_reg = (b_sm[(kk) * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn *
                                      BLOCK_DIM_N / VEC_DIM_N]);
                    }

                    c_reg[mm * VEC_DIM_N + nn] += a_reg * b_reg;
                }
            }
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int BLOCK_DIM_M>
    __device__ void compute_tile_attention_softmax(const int N, const int thread_x,
                                                   const int warp_id,
                                                   const float scale,
                                                   const int start_m,
                                                   const int start_n,
                                                   float *c_reg,
                                                   float *max_value_global,
                                                   float *sum_value_global,
                                                   float *max_value,
                                                   float *sum_value,
                                                   float *new_max,
                                                   float *new_sum) {
        const int base_n = start_n + thread_x;
        //#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            float old_max = max_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
            float old_sum = sum_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];

            //#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                if ((nn * 32 + base_n) < N) {
                    c_reg[mm * VEC_DIM_N + nn] *= scale;
                    max_value[mm] = fmaxf(max_value[mm], c_reg[mm * VEC_DIM_N + nn]);
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                max_value[mm] = fmaxf(max_value[mm], __shfl_xor_sync(0xffffffff, max_value[mm], offset, 32));
            }

            //#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                if ((nn * 32 + base_n) < N) {
                    c_reg[mm * VEC_DIM_N + nn] = expf(c_reg[mm * VEC_DIM_N + nn] - max_value[mm]);
                    sum_value[mm] += c_reg[mm * VEC_DIM_N + nn];
                }
            }

#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum_value[mm] += __shfl_xor_sync(0xffffffff, sum_value[mm], offset, 32);
            }
            new_max[mm] = fmaxf(max_value[mm], old_max);
            new_sum[mm] = sum_value[mm] * expf(max_value[mm] - new_max[mm]) + old_sum * expf(old_max - new_max[mm]);
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __device__ void compute_tile_attention_pv(const int N, const int v_ld,
                                              const int thread_x,
                                              const int warp_id,
                                              const int start_m,
                                              const int start_n,
                                              const T *v_sm,
                                              float *c_reg,
                                              float *old_max_value,
                                              float *old_sum_value,
                                              float *current_max_value,
                                              float *current_sum_value,
                                              float *new_max_value,
                                              float *new_sum_value,
                                              float *output) {
        const int base_n = start_n + thread_x;
        for (int kk = 0; kk < BLOCK_DIM_K; kk++) {
            //#pragma unroll
            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                float old_max = old_max_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
                float old_sum = old_sum_value[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
                float acc = 0.0f;
                //#pragma unroll
                for (int nn = 0; nn < VEC_DIM_N; nn++) {
                    float v_reg = 0.0f;
                    if constexpr (std::is_same_v<T, half>) {
                        v_reg = __half2float(
                            v_sm[kk * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                    } else if constexpr (std::is_same_v<T, float>) {
                        v_reg = v_sm[kk * (BLOCK_DIM_N + PAD_SIZE) + thread_x + nn * BLOCK_DIM_N / VEC_DIM_N];
                    }
                    if ((nn * 32 + base_n) < N) {
                        acc += v_reg * c_reg[mm * VEC_DIM_N + nn];
                    }
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
                        output[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] * expf(
                            old_max - new_max_value[mm]) +
                        acc * expf(current_max_value[mm] - new_max_value[mm]);
            }
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE>
    __global__ void flash_attention_cuda(const int M, const int N, const int D,
                                         const int q_ld, const int k_ld, const int v_ld,
                                         const float scale,
                                         const int start_n,
                                         const int n_block_size,
                                         const T *__restrict__ q_global,
                                         const T *__restrict__ k_global,
                                         const T *__restrict__ v_global,
                                         float *__restrict__ max_value_global,
                                         float *__restrict__ sum_value_global,
                                         float *out_put) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int ld_thread_block_n = BLOCK_DIM_K / VEC_DIM_K;
        const int thread_x = thread_id % ld_thread_block_n;
        const int warp_id = thread_id / ld_thread_block_n;
        //const int warp_group_id = thread_id / 128;

        const int block_x = blockIdx.x;
        const int start_m = block_x * BLOCK_DIM_M;

        __shared__ T sm[2][BLOCK_DIM_K][BLOCK_DIM_M + PAD_SIZE];
        float c_reg[VEC_DIM_M * VEC_DIM_N] = {0};

        for (int d = 0; d < D; d += BLOCK_DIM_K) {
            //加载当前q块;
            load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
                q_ld, M, thread_x, warp_id, start_m, d,
                q_global, &sm[0][0][0]);
            load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
                k_ld, N, thread_x, warp_id, start_n, d,
                k_global, &sm[1][0][0]);

            __syncthreads();


            const int k_size = min(BLOCK_DIM_K, D - d);
            compute_tile_attention_gemm<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N, PAD_SIZE>(
                scale, k_size, thread_x, warp_id, &sm[0][0][0], &sm[1][0][0], c_reg);

            __syncthreads();

            load_tile_vec_row_major_n<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
                v_ld, N, thread_x, warp_id, start_n, d,
                v_global, &sm[1][0][0]);
            __syncthreads();

            const int base_n = start_n + thread_x;
            const int valid_n = min(VEC_DIM_N, (N - base_n + 31) / 32);

            for (int mm = 0; mm < VEC_DIM_M; mm++) {
                float old_max = max_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
                float old_sum = sum_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M];
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

                for (int kk = d; kk < BLOCK_DIM_K; kk++) {
                    float acc = 0.0f;
                    for (int nn = 0; nn < valid_n; nn++) {
                        float v_reg = __half2float(
                            sm[1][kk][thread_x + nn * BLOCK_DIM_N / VEC_DIM_N]);
                        acc += v_reg * c_reg[mm * VEC_DIM_N + nn];
                    }
#pragma unroll
                    for (int offset = 16; offset > 0; offset /= 2) {
                        acc += __shfl_xor_sync(0xffffffff, acc, offset, 32);
                    }

                    out_put[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] =
                    (out_put[(start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M) * v_ld + kk] * old_sum * expf(
                         old_max - new_max_value) +
                     acc * expf(max_value - new_max_value)) / new_sum_value;
                }
                max_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_max_value;
                sum_value_global[start_m + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M] = new_sum_value;
            }
        }
    }

    //
    template<typename T>
    void tff::kernel::FlashAttn<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        //auto name = para_ptr->get_param<std::string>(0);
        tff::log::Logger::info("layer node, op:%s compute!", tff::kernel::FlashAttn<T>::get_op_name().c_str());

        constexpr int ELEMENTS_PER_LOAD = 2;
        constexpr int WARP_SIZE = 32;
        constexpr int THREAD_BLOCK_SIZE = 256;
        constexpr int BLOCK_DIM_K = 64;
        constexpr int VEC_DIM_N = 2;
        constexpr int VEC_DIM_K = 2;
        constexpr int VEC_DIM_M = 8;
        constexpr int BLOCK_DIM_M = 64; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_M;
        constexpr int BLOCK_DIM_N = 64; //THREAD_BLOCK_SIZE / (BLOCK_DIM_K / VEC_DIM_K) * VEC_DIM_N;
        constexpr int PAD_SIZE = 1; //BLOCK_DIM_K;


    }
    template<typename T>
    std::string tff::kernel::FlashAttn<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA;

        return name;
    }
    template class tff::kernel::FlashAttn<float>;
    template class tff::kernel::FlashAttn<double>;
    template class tff::kernel::FlashAttn<int32_t>;
    template class tff::kernel::FlashAttn<int64_t>;
    REGISTER_OP_OBJECT(FlashAttn, float);

    REGISTER_OP_OBJECT(FlashAttn, double);

    REGISTER_OP_OBJECT(FlashAttn, int32_t);

    REGISTER_OP_OBJECT(FlashAttn, int64_t);
}
