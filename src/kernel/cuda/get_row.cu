//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "core/runtime/KVCache.h"

namespace tff::kernel {
    template<typename T, const int WARP_SIZE, const int BLOCK_ROW_SIZE, const int BLOCK_COL_SIZE>
    __global__ __forceinline__ void get_value_by_table(
        const int64_t *__restrict__ block_table,
        T *__restrict__ dst,
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
                const int dst_row = start_offset + dst_row_base;
                if (dst_row_base < token_num) {
                    auto *dst_ptr = reinterpret_cast<half2 *>(dst_block_ptr);
                    auto *src_ptr = reinterpret_cast<half2 *>(block_table[row * dim1 + head_index]);
                    dst_ptr[dst_row * dim1 * dim0 / 2 + col] = src_ptr[thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL];
                    // if (thread_x == 0 && warp_id == 0 && head_index == 1) {
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
        const int g_dst_index = batch_index * dim2 * dim1 * dst_dim0 + block_index * BLOCK_SIZE * dst_dim0 +
                                start_offset *
                                dim1 * dst_dim0;
        const int g_src_index = batch_index * dim2 * dim1 * src_dim0 + block_index * BLOCK_SIZE * src_dim0 + start_row *
                                dim1 * src_dim0;

        auto *dst_block_ptr = dst + g_dst_index;
        auto *src_block_ptr = src + g_src_index;
        if ((g_dst_index + thread_id * dst_dim0) >= (start_offset *
                                                     dim1 * dst_dim0 + dim3 * dim1 * token_num * dst_dim0)) {
            return;
        }
        if ((g_src_index + thread_id * src_dim0) >= (start_row *
                                                     dim1 * src_dim0 + dim3 * dim1 * dim2 * src_dim0)) {
            return;
        }
        dst_block_ptr[thread_id * dst_dim0] = reinterpret_cast<int64_t>(&src_block_ptr[thread_id * src_dim0]);
        if (block_index == 0 && start_row == 0) {
            printf("thread_id: %d,block_index: %d, g_dst_index: %d, g_src_index : %d src[%d]: %lf, src[%d]: %lld \n",
                   thread_id, block_index, g_dst_index + thread_id * dst_dim0, g_src_index + thread_id * src_dim0,
                   thread_id * src_dim0,
                   __half2float(src_block_ptr[thread_id * src_dim0]),
                   thread_id * src_dim0,
                   reinterpret_cast<int64_t>(&src_block_ptr[thread_id * src_dim0]));
        }
    }

    template<typename T>
    void get_of_rows(const int seq_id, const int layer_id,
                     std::shared_ptr<tff::core::runtime::LLMKVCache> &kv_cache_ctx,
                     std::shared_ptr<core::memory::Tensor> &output_tensor,
                     std::shared_ptr<core::device::DeviceStream> &stream) {
        const int dim0 = output_tensor->get_shape()[0];
        const int dim1 = output_tensor->get_shape()[1];
        const int dim2 = output_tensor->get_shape()[2];
        auto batch = output_tensor->get_shape()[3];
        const int total_token_num = kv_cache_ctx->get_kv_token_num(seq_id, layer_id);
        const int pre_token_num = total_token_num - dim2;
        for (int i = 0; i < dim2; i += PAGE_SIZE) {
            constexpr int BLOCK_SIZE = PAGE_SIZE;
            const auto &[page_id, offset] = kv_cache_ctx->get_location(seq_id, layer_id, i);
            int token_num = PAGE_SIZE;
            if ((i + PAGE_SIZE) > dim2) {
                token_num = dim2 % PAGE_SIZE;
            }

            dim3 grid((PAGE_SIZE * dim1 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
            dim3 block(PAGE_SIZE, 1);
            if (output_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
                auto cache_tensor = kv_cache_ctx->get_k(seq_id, layer_id, page_id);
                if (cache_tensor == nullptr || cache_tensor->get_buffer() == nullptr) {
                    continue;
                }
                get_of_rows_kernel<BLOCK_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                    get_native_stream())>>>(static_cast<half *>(cache_tensor->get_buffer()->ptr()),
                                            static_cast<T *>(output_tensor->get_buffer()->ptr()),
                                            cache_tensor->get_shape()[0],
                                            dim0, dim1, dim2, batch, i, offset, token_num);
            } else if (output_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                auto cache_tensor = kv_cache_ctx->get_v(seq_id, layer_id, page_id);
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
    void tff::kernel::GetRow<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto kv_cache_ctx = kernel::base::get_param_value<std::shared_ptr<
            tff::core::runtime::LLMKVCache> >(0, para_ptr);
        auto seq_id = kernel::base::get_param_value<int>(1, para_ptr);
        auto layer_id = kernel::base::get_param_value<int>(2, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(3, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            para_ptr->get_param_count() - 1, para_ptr);
        if (output_tensor == nullptr || output_tensor->get_buffer() == nullptr) {
            return;
        }
        get_of_rows<T>(seq_id, layer_id, kv_cache_ctx, output_tensor, stream);
#ifdef _DEBUG
        stream->synchronize();
        int d = 128;
        int h = 8;
        int s = 36;
        const int cache_s = PAGE_SIZE * ((s + PAGE_SIZE - 1) / PAGE_SIZE);
        half *block_table_value_result = nullptr;
        cudaMalloc((void **) &block_table_value_result, d * h * cache_s * sizeof(half));


        for (int i = 0; i < s; i += PAGE_SIZE) {
            constexpr int BLOCK_COL_SIZE = 2 * PAGE_SIZE;
            constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;

            int token_num = PAGE_SIZE;
            if ((i + PAGE_SIZE) > s) {
                token_num = s % PAGE_SIZE;
            }
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            dim3 grid(h, (PAGE_SIZE + BLOCK_ROW_SIZE - 1) / BLOCK_ROW_SIZE, 1);
            dim3 block(32, 8);
            get_value_by_table<half, 32, BLOCK_ROW_SIZE, BLOCK_COL_SIZE><<<grid, block>>>(static_cast<int64_t*>(output_tensor->get_buffer()->ptr()),
                block_table_value_result,
                d, h, s, i, i, token_num);

            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);


            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            printf("Kernel time: %.4f ms\n", milliseconds);

        }
        std::array<int64_t, MAX_TENSOR_DIM> shape = {128, 8, 36, 1};
        auto tensor = std::make_shared<core::memory::Tensor>(core::memory::DataType::TFF_DATA_TYPE_F16,
            core::memory::MemoryType::TFF_MEM_TYPE_WORKSPACE, shape);
        tensor->set_buffer_data(block_table_value_result, tensor->get_bytes());
        tensor->set_allocator(output_tensor->get_allocator());
        const auto &name = kernel::base::get_param_value<std::string>(para_ptr->get_param_count() - 5, para_ptr);
        std::string filename = "";
        if (name == "k_cache_load_node") {
            filename = "Kcur_normed-0_result.ggml";
            varify(filename, tensor);
        } else if (name == "v_cache_load_node") {
            filename = "Vcur-0_result.ggml";
            varify(filename, tensor);
        } else if (name == "blk.0.attn_norm_rms_norm") {
            filename = "attn_norm-0_result.ggml";
            varify(filename, tensor);
        }
        cudaFree(block_table_value_result);
        block_table_value_result = nullptr;
#endif
    }


    template class tff::kernel::GetRow<int64_t>;
    REGISTER_OP_OBJECT(GetRow, int64_t);
}
