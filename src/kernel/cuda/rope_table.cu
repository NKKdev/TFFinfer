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
                               std::shared_ptr<core::device::DeviceStream> &stream) {
        constexpr int BLOCK_DIM_M = 64;
        constexpr int BLOCK_DIM_N = 32;
        constexpr int VEC_DIM_M = 8;
        constexpr int VEC_DIM_N = 1;
        constexpr int THREAD_BLOCK_DIM = (BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_N / VEC_DIM_N);
        dim3 grid((dim / 2 + BLOCK_DIM_N - 1) / BLOCK_DIM_N, (max_seq_len + BLOCK_DIM_M - 1) / BLOCK_DIM_M);
        dim3 block(THREAD_BLOCK_DIM, 1);

        float log_base = std::log(base);
        precompute_rope_table<T, VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N><<<grid, block, 0, static_cast<
            cudaStream_t>(stream->get_native_stream())>>>(max_seq_len, dim,
                                                          log_base, static_cast<T *>(out_table->get_buffer()->ptr()));
    }

    template<typename T>
    void tff::kernel::PreRopeTable<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto max_seq_len = kernel::base::get_param_value<const uint32_t>(0, para_ptr);
        auto dim = kernel::base::get_param_value<const uint32_t>(1, para_ptr);
        auto base = kernel::base::get_param_value<float>(2, para_ptr);
        auto scale = kernel::base::get_param_value<float>(3, para_ptr); //todo 支持rope_scale；
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            4, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        para_ptr->get_param_count() - 1, para_ptr);

        if (output_tensors == nullptr) {
            tff::log::Logger::error("Output tensor size mismatch");
            return;
        }

        precompute_rope_table<T>(max_seq_len, dim, base, output_tensors,stream);
    }


    template class tff::kernel::PreRopeTable<float>;
    REGISTER_OP_OBJECT(PreRopeTable, float);
}
