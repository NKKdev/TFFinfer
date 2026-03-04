//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "core/runtime/KVCache.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {
     template<typename T, const int WARP_SIZE, const int BLOCK_ROW_SIZE, const int BLOCK_COL_SIZE>
    __global__ __forceinline__ void get_value_by_table(
        const T *__restrict__ block_table,
        half *__restrict__ dst,
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
        const int VEC_DIM_ROW = BLOCK_ROW_SIZE / blockDim.y; // 4
        const int VEC_DIM_COL = BLOCK_COL_SIZE / blockDim.x; // 1

        const int src_row_base = start_row + block_row_index;
        const int src_col_base = head_index * dim0 / 2;
        const int g_index = batch_index * dim2 * dim1 * dim0;
        const auto dst_block_ptr = dst + g_index;

#pragma unroll
        for (int i = 0; i < VEC_DIM_ROW; i++) {
            const int row = src_row_base + warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
            if (row >= dim2) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < VEC_DIM_COL; j++) {
                const int col = src_col_base + thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL;
                if (col >= dim1 * dim0) {
                    continue;
                }
                const int dst_row_base = warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
                const int dst_row = block_row_index + dst_row_base;
                if (dst_row < dim2) {
                    auto *dst_ptr = reinterpret_cast<half2 *>(dst_block_ptr);
                    auto *src_ptr = reinterpret_cast<half2 *>(block_table[row * dim1 + head_index]);
                    dst_ptr[dst_row * dim1 * dim0 / 2 + col] = src_ptr[thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL];
                    // if (block_row_index == 32 && thread_x == 0 && warp_id == 0 && head_index == 0) {
                    //     printf("thread_x: %d, warp_id:%d, block_table[%d]: %lld, src_ptr[%d].x: %lf, src_ptr[%d].y: %lf "
                    //            "dst[%d].x: %lf, dst[%d].y: %lf\n",
                    //         thread_x, warp_id,
                    //         row * dim1 + head_index,
                    //         block_table[row * dim1 + head_index],
                    //         thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL,
                    //         __half2float(src_ptr[thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL].x),
                    //         col,
                    //         __half2float(src_ptr[thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL].y),
                    //         dst_row * dim1 * dim0 / 2 + col,
                    //         __half2float(dst_ptr[dst_row * dim1 * dim0 / 2 + col].x),
                    //         dst_row * dim1 * dim0 / 2 + col,
                    //         __half2float(dst_ptr[dst_row * dim1 * dim0 / 2 + col].y));
                    // }
                }
            }
        }
    }
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
        const int VEC_DIM_ROW = BLOCK_ROW_SIZE / blockDim.y; // 4
        const int VEC_DIM_COL = BLOCK_COL_SIZE / blockDim.x; // 1

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
                if (col >= dim1 * dim0) {
                    continue;
                }
                const int dst_row_base = warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
                const int dst_row = start_offset + dst_row_base;
                if (dst_row_base < token_num) {
                    if (std::is_same_v<T, float>) {
                        auto *dst_ptr = reinterpret_cast<float2 *>(dst_block_ptr);
                        dst_ptr[dst_row * dim1 * dim0 / 2 + col] = src_ptr[row * dim1 * dim0 / 2 + col];
                    } else if (std::is_same_v<T, half>) {
                        auto *dst_ptr = reinterpret_cast<half2 *>(dst_block_ptr);
                        dst_ptr[dst_row * dim1 * dim0 / 2 + col] = __float22half2_rn(
                            src_ptr[row * dim1 * dim0 / 2 + col]);
                    }
                }
            }
        }
    }
#ifdef _DEBUG
    template<typename T>
    void set_of_rows_cpu(const float *__restrict__ src, T *__restrict__ dst,
                         const int dim0,
                         const int dim1,
                         const int dim2,
                         const int start_row,
                         const int start_offset,
                         const int token_num) {
        const int src_row_base = start_row;
        for (int ss = 0; ss < 32; ++ss) {
            const int row = src_row_base + ss;
            if (row >= dim2) {
                continue;
            }
            for (int hh = 0; hh < dim1; ++hh) {
                const int src_col_base = hh * dim0;
                for (int dd = 0; dd < dim0; ++dd) {
                    if ((ss) >= token_num || (src_col_base + dd) >= dim1 * dim0) {
                        continue;
                    }
                    if (std::is_same_v<T, half>) {
                        dst[(start_offset + ss) * dim1 * dim0 + src_col_base + dd] =
                                __float2half_rn(src[row * dim1 * dim0 + src_col_base + dd]);
                    } else if (std::is_same_v<T, float>) {
                        dst[(start_offset + ss) * dim1 * dim0 + src_col_base + dd] =
                                (src[row * dim1 * dim0 + src_col_base + dd]);
                    }
                }
            }
        }
    }
#endif
    template<typename T>
    void set_of_rows(const int seq_id, const int layer_id,
                     std::shared_ptr<tff::core::runtime::LLMKVCache> &kv_cache_ctx,
                     std::shared_ptr<core::memory::Tensor> &input_tensor,
                     std::shared_ptr<core::device::DeviceStream> &stream,
                     std::shared_ptr<core::device::DeviceEvent> &event) {
        if (input_tensor == nullptr) {
            return;
        }
        const auto dim0 = input_tensor->get_shape()[0];
        const auto dim1 = input_tensor->get_shape()[1];
        const auto dim2 = input_tensor->get_shape()[2];
        const auto batch = input_tensor->get_shape()[3];

        const int total_token_num = kv_cache_ctx->get_kv_token_num(seq_id, layer_id);
        const int pre_token_num = total_token_num - dim2;
        if (pre_token_num < 0) {
            log::Logger::error("invalid kv cache");
            return;
        }
        for (int i = 0; i < dim2; i += PAGE_SIZE) {
            constexpr int BLOCK_COL_SIZE = 2 * PAGE_SIZE;
            constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;

            const auto &[page_id, offset] =
                    kv_cache_ctx->get_location(seq_id, layer_id, pre_token_num + i);

            int token_num = PAGE_SIZE;
            if ((i + PAGE_SIZE) > dim2) {
                token_num = dim2 % PAGE_SIZE;
            }

            dim3 grid(dim1, (PAGE_SIZE + BLOCK_ROW_SIZE - 1) / BLOCK_ROW_SIZE, batch);
            dim3 block(32, 8);
            if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K_NORM) {
                auto cache_tensor = kv_cache_ctx->get_k(seq_id, layer_id, page_id, event);
                if (cache_tensor == nullptr) {
                    tff::log::Logger::error("k cache is invalid");
                    continue;
                }
                if (cache_tensor->get_buffer() == nullptr) {
                    tff::log::Logger::error("k cache buffer is invalid");
                    continue;
                }
                set_of_rows_kernel<T, 32, BLOCK_ROW_SIZE, BLOCK_COL_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(
                    stream->
                    get_native_stream())>>>(static_cast<float *>(input_tensor->get_buffer()->ptr()),
                                            static_cast<T *>(cache_tensor->get_buffer()->ptr()),
                                            dim0, dim1, dim2, i, offset, token_num);
            } else if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                auto cache_tensor = kv_cache_ctx->get_v(seq_id, layer_id, page_id, event);
                if (cache_tensor == nullptr) {
                    tff::log::Logger::error("v cache is invalid");
                    continue;
                }
                if (cache_tensor->get_buffer() == nullptr) {
                    tff::log::Logger::error("v cache buffer is invalid");
                    continue;
                }
                set_of_rows_kernel<T, 32, BLOCK_ROW_SIZE, BLOCK_COL_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(
                    stream->
                    get_native_stream())>>>(static_cast<float *>(input_tensor->get_buffer()->ptr()),
                                            static_cast<T *>(cache_tensor->get_buffer()->ptr()),
                                            dim0, dim1, dim2, i, offset, token_num);
            }
        }
    }

    template<typename T>
    class SetRow<T,
                core::device::GPUTag> : public base::OPCreatorBase<SetRow<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_SET_ROWS;
        }
    };

    template<typename T>
    void tff::kernel::SetRow<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto kv_cache_ctx = kernel::base::get_param_value<std::shared_ptr<
            tff::core::runtime::LLMKVCache> >(SetRowBuilder::Params::KVCacheCtx, para_ptr);
        auto seq_id = kernel::base::get_param_value<int>(SetRowBuilder::Params::SeqId, para_ptr);
        auto layer_id = kernel::base::get_param_value<int>(SetRowBuilder::Params::LayerId, para_ptr);
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(
            SetRowBuilder::Params::In, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(
            SetRowBuilder::Params::Out, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            kernel::builder::OpParamBuilderBase<SetRowBuilder>::CommonParams::Stream, para_ptr);
        auto event = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceEvent> >(
            kernel::builder::OpParamBuilderBase<SetRowBuilder>::CommonParams::Event, para_ptr);
        if (input_tensor == nullptr) {
            return;
        }
        set_of_rows<T>(seq_id, layer_id, kv_cache_ctx, input_tensor, stream,  event);
    }


    template class tff::kernel::SetRow<half, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(SetRow, half, core::device::GPUTag);
    template class tff::kernel::SetRow<float, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(SetRow, float, core::device::GPUTag);
}
