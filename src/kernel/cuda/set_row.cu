//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "core/runtime/KVCache.h"

namespace tff::kernel {
    template<typename T, const int WARP_SIZE, const int BLOCK_ROW_SIZE>
    __global__ __forceinline__ void set_of_rows_kernel(const T *__restrict__ src, T *__restrict__ dst,
                                                       const int64_t *kv_idx, const int n_stride, const int M) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        const int start_m = blockIdx.y * M * n_stride + blockIdx.x * BLOCK_ROW_SIZE;
        const int thread_x = thread_id % WARP_SIZE;
        const int warp_id = thread_id / WARP_SIZE;
        const int THREAD_BLOCK_SIZE = blockDim.x * blockDim.y;
        const int VEC_DIM_ROW = BLOCK_ROW_SIZE / (THREAD_BLOCK_SIZE / WARP_SIZE);
        const int VEC_DIM_COL = n_stride / WARP_SIZE;

        const int src_row_base = start_m + warp_id;
        const int src_col_base = thread_x;
#pragma unroll
        for (int i = 0; i < VEC_DIM_ROW; i++) {
            const int src_row = src_row_base + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
            if (src_row >= M) {
                continue;
            }
#pragma unroll
            for (int j = 0; j < VEC_DIM_COL; j++) {
                const int src_col = src_col_base + j * n_stride / VEC_DIM_COL;
                if (src_col >= n_stride) {
                    continue;
                }
                dst[kv_idx[(warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW)]
                    * n_stride + thread_x + j * n_stride / VEC_DIM_COL] =
                        src[src_row * n_stride + src_col];
            }
        }
    }

    template<typename T>
    void set_of_rows(const int seq_id, const int layer_id,
                     std::shared_ptr<tff::core::runtime::LLMKVCache> &kv_cache_ctx,
                     std::unordered_map<int, std::shared_ptr<core::memory::Tensor> > &kv_idx,
                     std::shared_ptr<core::memory::Tensor> &cur_k,
                     std::shared_ptr<core::memory::Tensor> &cur_v,
                     std::shared_ptr<core::device::DeviceStream> &stream) {
        if (cur_k == nullptr) {
            return;
        }
        auto batch = cur_k->get_shape()[3];

        constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;

        for (auto &kv_id: kv_idx) {
            auto row = kv_id.second->get_shape()[0];
            auto col = cur_k->get_shape()[0] * cur_k->get_shape()[1];
            dim3 grid((row + BLOCK_ROW_SIZE - 1) / BLOCK_ROW_SIZE, batch, 1);
            dim3 block(BLOCK_ROW_SIZE, 8);

            auto k_cache_tensor = kv_cache_ctx->get_k(seq_id, layer_id, kv_id.first);
            if (k_cache_tensor == nullptr || k_cache_tensor->get_buffer() == nullptr) {
                continue;
            }
            set_of_rows_kernel<T, 32, BLOCK_ROW_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                get_native_stream())>>>(static_cast<T *>(cur_k->get_buffer()->ptr()),
                                        static_cast<T *>(k_cache_tensor->get_buffer()->ptr()),
                                        static_cast<int64_t *>(kv_id.second->get_buffer()->ptr()),
                                        col, row);

            auto v_cache_tensor = kv_cache_ctx->get_v(seq_id, layer_id, kv_id.first);
            if (v_cache_tensor == nullptr || v_cache_tensor->get_buffer() == nullptr) {
                continue;
            }
            set_of_rows_kernel<T, 32, BLOCK_ROW_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                get_native_stream())>>>(static_cast<T *>(cur_v->get_buffer()->ptr()),
                                        static_cast<T *>(v_cache_tensor->get_buffer()->ptr()),
                                        static_cast<int64_t *>(kv_id.second->get_buffer()->ptr()),
                                        col, row);
        }
    }

    template<typename T>
    void tff::kernel::SetRow<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto kv_cache_ctx = kernel::base::get_param_value<std::shared_ptr<
            tff::core::runtime::LLMKVCache> >(0, para_ptr);
        auto kv_idx = kernel::base::get_param_value<std::unordered_map<int, std::shared_ptr<
            core::memory::Tensor> > >(1, para_ptr);
        auto seq_id = kernel::base::get_param_value<int>(2, para_ptr);
        auto layer_id = kernel::base::get_param_value<int>(3, para_ptr);
        auto cur_k = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(4, para_ptr);
        auto cur_v = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(5, para_ptr);
        auto out_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(6, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            para_ptr->get_param_count() - 1, para_ptr);
        if (cur_k == nullptr || cur_v == nullptr || out_tensor == nullptr) {
            return;
        }

        set_of_rows<T>(seq_id, layer_id, kv_cache_ctx, kv_idx, cur_k, cur_v, stream);
    }


    template class tff::kernel::SetRow<float>;
    template class tff::kernel::SetRow<half>;

    REGISTER_OP_OBJECT(SetRow, float);

    REGISTER_OP_OBJECT(SetRow, half);
}
