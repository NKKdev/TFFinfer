//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "core/runtime/KVCache.h"

namespace tff::kernel {
    template<typename T, const int WARP_SIZE, const int BLOCK_ROW_SIZE, const int BLOCK_COL_SIZE>
    __global__ __forceinline__ void set_of_rows_kernel(const float *__restrict__ src, T *__restrict__ dst,
                                                       const int dim0,
                                                       const int dim1,
                                                       const int dim2,
                                                       const int start_row,
                                                       const int start_offset,
                                                       const int token_num) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        const int head_index = blockIdx.x;
        const int batch_index = blockIdx.z;
        const int block_row_index = blockIdx.y * BLOCK_ROW_SIZE;

        const int thread_x = thread_id % WARP_SIZE;
        const int warp_id = thread_id / WARP_SIZE;
        const int VEC_DIM_ROW = BLOCK_ROW_SIZE / blockDim.y;// 4
        const int VEC_DIM_COL = BLOCK_COL_SIZE / blockDim.x;// 1
        const int src_row_base = start_row + block_row_index;
        const int src_col_base = head_index * dim0 / 2;
        const int g_index = batch_index * dim2 * dim1 * dim0;
        const auto src_block_ptr = src + g_index;
        const auto dst_block_ptr = dst + g_index;
        const auto *src_ptr = reinterpret_cast<const float2 *>(src_block_ptr);

#pragma unroll
        for (int i = 0; i < VEC_DIM_ROW; i++) {
            const int row = src_row_base + warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
            if (row >= dim2) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < VEC_DIM_COL; j++) {
                const int col = src_col_base + thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL;
                if (col >= dim0) {
                    continue;
                }
                const int dst_row = start_offset + warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
                if (dst_row < token_num) {
                    if (std::is_same_v<T, float>) {
                        auto *dst_ptr = reinterpret_cast<float2*>(dst_block_ptr);
                        dst_ptr[dst_row * dim1 * dim0 + col] = src_ptr[row * dim1 * dim0 + col];
                    }else if (std::is_same_v<T, half>) {
                        auto *dst_ptr = reinterpret_cast<half2*>(dst_block_ptr);
                        dst_ptr[dst_row * dim1 * dim0 + col] = __float22half2_rn(src_ptr[row * dim1 * dim0 + col]);
                    }
                }
            }
        }
    }

    template<typename T>
    void set_of_rows(const int seq_id, const int layer_id,
                     std::shared_ptr<tff::core::runtime::LLMKVCache> &kv_cache_ctx,
                     std::shared_ptr<core::memory::Tensor> &input_tensor,
                     std::shared_ptr<core::memory::Tensor> &output_tensor,
                     std::shared_ptr<core::device::DeviceStream> &stream) {
        if (input_tensor == nullptr || output_tensor == nullptr || output_tensor->get_buffer() == nullptr) {
            return;
        }
        *output_tensor = *input_tensor;
        const auto dim0  = input_tensor->get_shape()[0];
        const auto dim1  = input_tensor->get_shape()[1];
        const auto dim2  = input_tensor->get_shape()[2];
        const auto batch = input_tensor->get_shape()[3];

        const int total_token_num = kv_cache_ctx->get_kv_token_num(seq_id, layer_id);
        const int pre_token_num = total_token_num - dim2;
        for (int i = 0; i < dim2; i += PAGE_SIZE) {

            constexpr int BLOCK_COL_SIZE = PAGE_SIZE;
            constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;

            const auto &[page_id, offset] = kv_cache_ctx->get_location(seq_id, layer_id, i);

            int token_num = PAGE_SIZE;
            if ((i + PAGE_SIZE) > dim2) {
                token_num = dim2 % PAGE_SIZE;
            }

            dim3 grid(dim1, (PAGE_SIZE + BLOCK_ROW_SIZE - 1) / BLOCK_ROW_SIZE, batch);
            dim3 block(32, 8);
            if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
                auto cache_tensor = kv_cache_ctx->get_k(seq_id, layer_id, page_id);
                if (cache_tensor == nullptr) {
                    tff::log::Logger::error("k cache is invalid");
                    continue;
                }
                if (cache_tensor->get_buffer() == nullptr) {
                    tff::log::Logger::error("k cache buffer is invalid");
                    continue;
                }
                set_of_rows_kernel<T, 32, BLOCK_ROW_SIZE, BLOCK_COL_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                    get_native_stream())>>>(static_cast<float *>(input_tensor->get_buffer()->ptr()),
                                            static_cast<T *>(cache_tensor->get_buffer()->ptr()),
                                            dim0, dim1, dim2, i, offset, token_num);
            }else if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                auto cache_tensor = kv_cache_ctx->get_v(seq_id, layer_id, page_id);
                if (cache_tensor == nullptr) {
                    tff::log::Logger::error("v cache is invalid");
                    continue;
                }
                if (cache_tensor->get_buffer() == nullptr) {
                    tff::log::Logger::error("v cache buffer is invalid");
                    continue;
                }
                set_of_rows_kernel<T, 32, BLOCK_ROW_SIZE, BLOCK_COL_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                    get_native_stream())>>>(static_cast<float *>(input_tensor->get_buffer()->ptr()),
                                            static_cast<T *>(cache_tensor->get_buffer()->ptr()),
                                            dim0, dim1, dim2, i, offset, token_num);
            }

        }
    }

    template<typename T>
    void tff::kernel::SetRow<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto kv_cache_ctx = kernel::base::get_param_value<std::shared_ptr<
            tff::core::runtime::LLMKVCache> >(0, para_ptr);
        auto seq_id = kernel::base::get_param_value<int>(1, para_ptr);
        auto layer_id = kernel::base::get_param_value<int>(2, para_ptr);
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(3, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(4, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            para_ptr->get_param_count() - 1, para_ptr);
        if (input_tensor == nullptr || output_tensor == nullptr) {
            return;
        }

        set_of_rows<T>(seq_id, layer_id, kv_cache_ctx, input_tensor,output_tensor, stream);
    }


    template class tff::kernel::SetRow<float>;
    template class tff::kernel::SetRow<half>;

    REGISTER_OP_OBJECT(SetRow, float);

    REGISTER_OP_OBJECT(SetRow, half);
}
