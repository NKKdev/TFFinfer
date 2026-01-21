//
// Created by nkk on 2025/12/29.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {

    template<typename T, const int VEC_DIM_M, const int VEC_DIM_N, const int BLOCK_DIM_M, const int BLOCK_DIM_N>
    __global__ void precompute_rope_table(const int max_seq_len, const int dim, const float log_base,
                                          T *out_table) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % 32;
        const int warp_id = thread_id / 32;

        const int start_m = blockIdx.y * BLOCK_DIM_M;
        const int start_n = blockIdx.x * BLOCK_DIM_N;

        auto *out_ptr = reinterpret_cast<float2 *>(out_table);
#pragma unroll
        for (int mm = 0; mm < VEC_DIM_M; ++mm) {
            const int row_index = (start_m + warp_id + mm * (BLOCK_DIM_M / VEC_DIM_M));
            if (row_index >= max_seq_len) {
                continue;
            }
#pragma unroll
            for (int nn = 0; nn < VEC_DIM_N; ++nn) {
                const int col_index = (start_n + thread_x + nn * (BLOCK_DIM_N / VEC_DIM_N));
                if (col_index >= dim / 2) {
                    continue;
                }

                const float exponent = -static_cast<float>((2.0f * col_index) / static_cast<float>(dim) * log_base);
                const float value = row_index * exp(exponent);
                out_ptr[row_index * (dim / 2) + col_index].x = cos(value);
                out_ptr[row_index * (dim / 2) + col_index].y = sin(value);
            }
        }
    }
    template<typename T>
    void precompute_rope_table(const int max_seq_len, const int dim, const float base,
        std::shared_ptr<tff::core::memory::Tensor> &out_table,
        std::shared_ptr<core::runtime::LLMMemManager> &mem_buffer_manager_ptr) {

        constexpr int BLOCK_DIM_M = 64;
        constexpr int BLOCK_DIM_N = 32;
        constexpr int VEC_DIM_M = 8;
        constexpr int VEC_DIM_N = 1;
        constexpr int THREAD_BLOCK_DIM = (BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_N / VEC_DIM_N);
        dim3 grid((dim / 2 + BLOCK_DIM_N - 1) / BLOCK_DIM_N, (max_seq_len + BLOCK_DIM_M - 1) / BLOCK_DIM_M);
        dim3 block(THREAD_BLOCK_DIM, 1);

        float log_base = std::log(base);
        precompute_rope_table<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N><<<grid, block>>>(max_seq_len, dim,
            log_base, static_cast<T *>(out_table->get_buffer()->ptr()));
    }
    template<typename T>
    void tff::kernel::PreRopeTable<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), PreRopeTable<T>::get_op_name().c_str());
        auto max_seq_len = get_param_value<const uint32_t>(1, para_ptr);
        auto dim = get_param_value<const uint32_t>(2, para_ptr);
        auto base = get_param_value<float>(3, para_ptr);
        auto scale = get_param_value<float>(4, para_ptr);//todo 支持rope_scale；
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            5, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(6, para_ptr);
        if (output_tensors == nullptr) {
            tff::log::Logger::error("Output tensor size mismatch");
            return;
        }

        precompute_rope_table<T>(max_seq_len, dim, base, output_tensors, mem_buffer_manager_ptr);
    }

    template<typename T>
    std::string tff::kernel::PreRopeTable<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_PRE_ROPE_TABLE);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;
        return name;
    }
    template class tff::kernel::PreRopeTable<float>;
    REGISTER_OP_OBJECT(PreRopeTable, float);
}
