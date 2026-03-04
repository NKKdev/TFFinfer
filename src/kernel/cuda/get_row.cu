//
// Created by nkk on 2025/12/17.
//
#include "../../../../../../../../../usr/local/cuda/include/cuda/__ptx/ptx_dot_variants.h"
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "core/runtime/KVCache.h"

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

    template<const int BLOCK_SIZE>
    __global__ __forceinline__ void get_of_rows_kernel(const half *__restrict__ src,
                                                       int64_t *__restrict__ dst,
                                                       const int src_dim0,
                                                       const int dst_dim0,
                                                       const int dim1,
                                                       const int dim2,
                                                       const int dim3,
                                                       const int start_row,
                                                       const int start_offset,
                                                       const int token_num) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        const int batch_index = blockIdx.y;
        const int block_index = blockIdx.x;
        const int g_src_index = batch_index * dim2 * dim1 * src_dim0 + block_index * BLOCK_SIZE * src_dim0 +
                                start_offset * dim1 * src_dim0;
        const int g_dst_index = batch_index * dim2 * dim1 * dst_dim0 + block_index * BLOCK_SIZE * dst_dim0 + start_row *
                                dim1 * dst_dim0;

        auto *dst_block_ptr = dst + g_dst_index;
        auto *src_block_ptr = src + g_src_index;
        if ((g_src_index + thread_id * src_dim0) >= ((start_offset + token_num) * dim1 * src_dim0)) {
            return;
        }
        if ((g_dst_index + thread_id * dst_dim0) >= (dim1 * dim2 * dst_dim0)) {
            return;
        }
        dst_block_ptr[thread_id * dst_dim0] = reinterpret_cast<int64_t>(&src_block_ptr[thread_id * src_dim0]);
        // if (block_index == 0 && start_row == 32) {
        //     printf("thread_id: %d,block_index: %d, g_dst_index: %d, g_src_index : %d src[%d]: %lf, src[%d]: %lld \n",
        //            thread_id, block_index, g_dst_index + thread_id * dst_dim0, g_src_index + thread_id * src_dim0,
        //            thread_id * src_dim0,
        //            __half2float(src_block_ptr[thread_id * src_dim0]),
        //            thread_id * src_dim0,
        //            reinterpret_cast<int64_t>(&src_block_ptr[thread_id * src_dim0]));
        // }
    }
#ifdef _DEBUG
    template<typename T>
    void get_of_rows_cpu(const T *__restrict__ src, int64_t *__restrict__ dst,
                         const int src_dim0,
                         const int dst_dim0,
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
                const int src_col_base = hh * src_dim0;
                const int dst_col_base = hh * dst_dim0;
                if ((ss) >= token_num || (src_col_base) >= dim1 * src_dim0) {
                    continue;
                }
                dst[(start_offset + ss) * dim1 * dst_dim0 + dst_col_base] =
                        reinterpret_cast<int64_t>(&src[row * dim1 * src_dim0 + src_col_base]);
            }
        }
    }
#endif
    template<typename T>
    class GetRow<T,
                core::device::GPUTag> : public base::OPCreatorBase<GetRow<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_GET_ROWS;
        }
    };

    template<typename T>
    void get_of_rows(const int seq_id, const int layer_id,
                     std::shared_ptr<tff::core::runtime::LLMKVCache> &kv_cache_ctx,
                     std::shared_ptr<core::memory::Tensor> &output_tensor,
                     std::shared_ptr<core::device::DeviceStream> &stream,
                     std::shared_ptr<core::device::DeviceEvent> &event) {
        const int dim0 = output_tensor->get_shape()[0];
        const int dim1 = output_tensor->get_shape()[1];
        const int dim2 = output_tensor->get_shape()[2];
        auto batch = output_tensor->get_shape()[3];
        const int total_token_num = kv_cache_ctx->get_kv_token_num(seq_id, layer_id);
        for (int i = 0; i < total_token_num; i += PAGE_SIZE) {
            constexpr int BLOCK_SIZE = PAGE_SIZE;
            const auto &[page_id, offset] =
                kv_cache_ctx->get_location(seq_id, layer_id, i);
            int token_num = PAGE_SIZE;
            if ((i + PAGE_SIZE) > total_token_num) {
                token_num = total_token_num % PAGE_SIZE;
            }

            dim3 grid((PAGE_SIZE * dim1 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
            dim3 block(PAGE_SIZE, 1);
            if (output_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
                auto cache_tensor = kv_cache_ctx->get_k(seq_id, layer_id, page_id, event);
                if (cache_tensor == nullptr || cache_tensor->get_buffer() == nullptr) {
                    continue;
                }
                get_of_rows_kernel<BLOCK_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                    get_native_stream())>>>(static_cast<half *>(cache_tensor->get_buffer()->ptr()),
                                            static_cast<T *>(output_tensor->get_buffer()->ptr()),
                                            cache_tensor->get_shape()[0],
                                            dim0, dim1, dim2, batch, i, offset, token_num);

            } else if (output_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                auto cache_tensor = kv_cache_ctx->get_v(seq_id, layer_id, page_id, event);
                if (cache_tensor == nullptr || cache_tensor->get_buffer() == nullptr) {
                    continue;
                }
                get_of_rows_kernel<BLOCK_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                    get_native_stream())>>>(static_cast<half *>(cache_tensor->get_buffer()->ptr()),
                                            static_cast<T *>(output_tensor->get_buffer()->ptr()),
                                            cache_tensor->get_shape()[0],
                                            dim0, dim1, dim2, batch, i, offset, token_num);
            }
        }
    }

    template<typename T>
    void tff::kernel::GetRow<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto kv_cache_ctx = kernel::base::get_param_value<std::shared_ptr<
            tff::core::runtime::LLMKVCache> >(GetRowBuilder::Params::KVCacheCtx, para_ptr);
        auto seq_id = kernel::base::get_param_value<int>(GetRowBuilder::Params::SeqId, para_ptr);
        auto layer_id = kernel::base::get_param_value<int>(GetRowBuilder::Params::LayerId, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(GetRowBuilder::Params::Out, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            kernel::builder::OpParamBuilderBase<MemOptOPBuilder>::CommonParams::Stream, para_ptr);
        auto event = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceEvent> >(
            kernel::builder::OpParamBuilderBase<MemOptOPBuilder>::CommonParams::Event, para_ptr);
        if (output_tensor == nullptr || output_tensor->get_buffer() == nullptr) {
            return;
        }
        get_of_rows<T>(seq_id, layer_id, kv_cache_ctx, output_tensor, stream, event);
    }


    template class tff::kernel::GetRow<int64_t, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(GetRow, int64_t, core::device::GPUTag);
}
