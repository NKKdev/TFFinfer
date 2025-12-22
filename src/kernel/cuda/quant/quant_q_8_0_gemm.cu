//
// Created by nkk on 2025/12/22.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int PAD_SIZE>
    static __device__ void load_tile_quant_vec(const int ld, const int dim,
                                               const int thread_x, const int warp_id,
                                               const int start_block,
                                               const int k,
                                               const int *__restrict__ global_mem,
                                               int *sm) {
        constexpr int INTS_PER_INT4 = 4;
        constexpr int INT4S_PER_VEC_K = VEC_DIM_K / INTS_PER_INT4;
        const int4 *__restrict__ global_int4 = reinterpret_cast<const int4 *>(global_mem);
        const int vec_ld = ld * 8 / INTS_PER_INT4;

        for (int j = 0; j < VEC_DIM_LD; ++j) {
            int dim0 = start_block + warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD;
            for (int i = 0; i < INT4S_PER_VEC_K; ++i) {
                int dim1 = (k * INT4S_PER_VEC_K + thread_x + i * BLOCK_DIM_K / VEC_DIM_K);
                int4 val = make_int4(0, 0, 0, 0);
                if (dim1 < vec_ld && dim0 < dim) {
                    val = global_int4[dim0 * vec_ld + dim1];
                }
                sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K + PAD_SIZE) + thread_x * INTS_PER_INT4 + i *
                   BLOCK_DIM_K
                   / VEC_DIM_K * INTS_PER_INT4 + 0] =
                        val.x;
                sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K + PAD_SIZE) + thread_x * INTS_PER_INT4 + i *
                   BLOCK_DIM_K
                   / VEC_DIM_K * INTS_PER_INT4 + 1] =
                        val.y;
                sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K + PAD_SIZE) + thread_x * INTS_PER_INT4 + i *
                   BLOCK_DIM_K
                   / VEC_DIM_K * INTS_PER_INT4 + 2] =
                        val.z;
                sm[(warp_id + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K + PAD_SIZE) + thread_x * INTS_PER_INT4 + i *
                   BLOCK_DIM_K
                   / VEC_DIM_K * INTS_PER_INT4 + 3] =
                        val.w;
            }
        }
    }

    template<typename T, const int VEC_DIM_LD, const int VEC_DIM_K,
        const int BLOCK_DIM_LD, const int BLOCK_DIM_K, const int PAD_SIZE,
        const int QUANT_BLOCK_SIZE>
    static __device__ void load_tile_scale(const int ld, const int dim,
                                           const int thread_x, const int thread_y,
                                           const int start_block,
                                           const int k,
                                           const half *__restrict__ global_mem,
                                           half *sm) {
        for (int j = 0; j < VEC_DIM_LD; ++j) {
            int dim0 = start_block + thread_y + j * BLOCK_DIM_LD / VEC_DIM_LD;
            for (int i = 0; i < (VEC_DIM_K * sizeof(int)) / QUANT_BLOCK_SIZE; ++i) {
                int dim1 = k + thread_x + i * BLOCK_DIM_K / VEC_DIM_K;
                half val = 0;
                if (dim1 < ld && dim0 < dim) {
                    val = global_mem[dim0 * ld + dim1];
                }
                sm[(thread_y + j * BLOCK_DIM_LD / VEC_DIM_LD) * (BLOCK_DIM_K / VEC_DIM_K + PAD_SIZE) + thread_x + i *
                   BLOCK_DIM_K /
                   VEC_DIM_K]
                        = val;
            }
        }
    }

    static __device__ int vec_dot_product(int a, int b, int c_sum) {
        return __dp4a(a, b, c_sum);
    }

    template<const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE,
        const int VEC_DOT_PRODUCT>
    static __device__ void compute_tile(const int thread_x, const int thread_y,
                                        half *a_scale_sm,
                                        half *b_scale_sm,
                                        int *a_quant_sm,
                                        int *b_quant_sm,
                                        float *c_reg) {
        //#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            //#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; nn++) {
                float sum = 0;
#pragma unroll
                for (int kk = 0; kk < BLOCK_DIM_K / VEC_DIM_K; kk++) {
                    float a_scale[VEC_DIM_M];
                    float b_scale[VEC_DIM_N];
                    a_scale[mm] = __half2float(
                        a_scale_sm[(thread_y + mm * BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_K / VEC_DIM_K + PAD_SIZE) +
                                   kk]);
                    b_scale[nn] = __half2float(
                        b_scale_sm[(thread_x + nn * BLOCK_DIM_N / VEC_DIM_N) * (BLOCK_DIM_K / VEC_DIM_K + PAD_SIZE) +
                                   kk]);
                    int block_sum = 0;
#pragma unroll
                    for (int kk_q = 0; kk_q < VEC_DOT_PRODUCT; kk_q++) {
                        const int kk_index = kk * VEC_DOT_PRODUCT + kk_q;
                        int a_reg[VEC_DIM_M] = {0};
                        int b_reg[VEC_DIM_N] = {0};
                        a_reg[mm] = a_quant_sm[(thread_y + mm * BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_K + PAD_SIZE) +
                                               kk_index];
                        b_reg[nn] = b_quant_sm[(thread_x + nn * BLOCK_DIM_N / VEC_DIM_N) * (BLOCK_DIM_K + PAD_SIZE) +
                                               kk_index];

                        block_sum = vec_dot_product(a_reg[mm], b_reg[nn], block_sum);
                    }
                    sum += (block_sum) * (a_scale[mm]) * (b_scale[nn]);
                }

                c_reg[mm * VEC_DIM_N + nn] += sum;
            }
        }
    }

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N,
        const int VEC_DIM_K, const int BLOCK_DIM_M, const int BLOCK_DIM_N, const int BLOCK_DIM_K, const int PAD_SIZE,
        const int QUANT_BLOCK_SIZE, const int VEC_DOT_PRODUCT>
    __global__ void gemm_quant_q_8_0_nt_reshape(
        const int M, const int N, const int K,
        const int a_ld, const int b_ld, const int c_ld,
        const half *__restrict__ a_scale,
        const half *__restrict__ b_scale,
        const int32_t *__restrict__ a_quant_data,
        const int32_t *__restrict__ b_quant_data,
        T *__restrict__ c) {
        const int g_thread_id = threadIdx.x + threadIdx.y * blockDim.x;

        const int thread_y = g_thread_id / (BLOCK_DIM_N / VEC_DIM_N);
        const int thread_x = g_thread_id % (BLOCK_DIM_N / VEC_DIM_N);

        const int start_m = blockIdx.y * BLOCK_DIM_M;
        const int start_n = blockIdx.x * BLOCK_DIM_N;


        __shared__ int a_quant_data_sm[BLOCK_DIM_M][BLOCK_DIM_K + PAD_SIZE];
        __shared__ int b_quant_data_sm[BLOCK_DIM_N][BLOCK_DIM_K + PAD_SIZE];
        __shared__ half a_scale_data_sm[BLOCK_DIM_K][BLOCK_DIM_K / VEC_DIM_K + PAD_SIZE];
        __shared__ half b_scale_data_sm[BLOCK_DIM_N][BLOCK_DIM_K / VEC_DIM_K + PAD_SIZE];
        float c_reg[VEC_DIM_M][VEC_DIM_N] = {0};

        for (size_t k = 0; k < K; k += BLOCK_DIM_K / VEC_DIM_K) {
            load_tile_quant_vec<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
                a_ld, M, thread_x, thread_y, start_m, k, a_quant_data,
                &a_quant_data_sm[0][0]);
            load_tile_quant_vec<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE>(
                b_ld, N, thread_x, thread_y, start_n, k, b_quant_data,
                &b_quant_data_sm[0][0]);
            load_tile_scale<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, QUANT_BLOCK_SIZE>(
                a_ld, M, thread_x, thread_y, start_m, k, a_scale,
                &a_scale_data_sm[0][0]);
            load_tile_scale<T, VEC_DIM_M, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_K, PAD_SIZE, QUANT_BLOCK_SIZE>(
                b_ld, N, thread_x, thread_y, start_n, k, b_scale,
                &b_scale_data_sm[0][0]);
            __syncthreads();

            compute_tile<VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_DIM_M, BLOCK_DIM_N, BLOCK_DIM_K, PAD_SIZE,
                VEC_DOT_PRODUCT>(thread_x, thread_y,
                                 &a_scale_data_sm[0][0], &b_scale_data_sm[0][0],
                                 &a_quant_data_sm[0][0], &b_quant_data_sm[0][0], &c_reg[0][0]);
            __syncthreads();
        }

        for (int mm = 0; mm < VEC_DIM_M; mm++) {
            const int dst_m_index = start_m + thread_y + mm * BLOCK_DIM_M / VEC_DIM_M;
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
    void quant_q_8_0_matmul(const int quant_block_size, const int M, const int N, const int K,
                            void *quant_a,
                            void *quant_b,
                            T *c) {
        if (std::is_same_v<T, half>) {
            constexpr int QUANT_BLOCK_SIZE = 32;
            constexpr int VEC_DIM_M = 2;
            constexpr int VEC_DIM_N = 2;
            constexpr int VEC_DIM_K = 8;
            constexpr int BLOCK_M_DIM = 32;
            constexpr int BLOCK_N_DIM = 32;
            constexpr int BLOCK_K_DIM = 128;
            constexpr int PAD_SIZE = 1;
            constexpr int VEC_DOT_PRODUCT = QUANT_BLOCK_SIZE / sizeof(int);

            auto scale_a = static_cast<half *>(quant_a);
            auto scale_b = static_cast<half *>(quant_b);
            auto quant_a_data = static_cast<int32_t *>(quant_a + M * K / quant_block_size);
            auto quant_b_data = static_cast<int32_t *>(quant_b + N * K / quant_block_size);

            dim3 grid((N + BLOCK_N_DIM - 1) / BLOCK_N_DIM, (M + BLOCK_M_DIM - 1) / BLOCK_M_DIM, 1);
            dim3 block(BLOCK_N_DIM / VEC_DIM_N, BLOCK_M_DIM / VEC_DIM_M, 1);
            gemm_quant_q_8_0_nt_reshape<T, VEC_DIM_M, VEC_DIM_N, VEC_DIM_K, BLOCK_M_DIM, BLOCK_N_DIM, BLOCK_K_DIM,
                PAD_SIZE, QUANT_BLOCK_SIZE, VEC_DOT_PRODUCT><<<grid, block>>>(
                M, N, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, K / QUANT_BLOCK_SIZE, N, scale_a, scale_b,
                quant_a_data,
                quant_b_data,
                 c);
        } else if (std::is_same_v<T, float>) {
            //todo float impl
        }
    }

    template<typename T>
    void tff::kernel::QuantQ8MatMul<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), QuantQ8MatMul<T>::get_op_name().c_str());
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            1, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            2, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(3, para_ptr);

        if (input_tensors.size() != 2) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        if (output_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        auto weight_tensor = input_tensors.at(0);
        if (weight_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("weight tensor is null!");
            return;
        }
        auto x_tensor = input_tensors.at(1);
        if (x_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("x tensor is null!");
            return;
        }
        auto output = output_tensors.at(0);
        if (output->get_buffer() == nullptr) {
            auto mem_buffer_pair = mem_buffer_manager_ptr->get_gpu_memory();
            if (mem_buffer_pair.second == nullptr) {
                tff::log::Logger::error("there is no GPU memory!");
                return;
            }
            output->set_buffer_data(mem_buffer_pair.second, output->get_bytes(), mem_buffer_pair.first);
        }
        const int K = weight_tensor->get_shape()[0];
        const int M = weight_tensor->get_shape()[1];
        const int B = weight_tensor->get_shape()[2];//todo impl batches
        const int N = x_tensor->get_shape()[1];
        //
        quant_q_8_0_matmul<T>(tff::core::quant::Q_8_0::BLOCK_SIZE, M, N, K,weight_tensor->get_buffer()->ptr(),
            x_tensor->get_buffer()->ptr(), static_cast<T*>(output->get_buffer()->ptr()));

        mem_buffer_manager_ptr->reset_gpu_memory(weight_tensor->get_external_memory_index());
        mem_buffer_manager_ptr->reset_gpu_memory(x_tensor->get_external_memory_index());
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
