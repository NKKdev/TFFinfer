//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "core/runtime/KVCache.h"

namespace tff::kernel {
    template<typename T>
    void get_of_row(const int seq_id, const int layer_id,
                    std::shared_ptr<tff::core::runtime::LLMKVCache> &kv_cache_ctx,
                    std::unordered_map<int, std::shared_ptr<core::memory::Tensor> > &kv_idx,
                    std::shared_ptr<core::memory::Tensor> &output_tensor,
                    std::shared_ptr<core::device::DeviceStream> &stream) {
        constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;
        const int n_stride = output_tensor->get_shape()[0];
        T *output = static_cast<T *>(output_tensor->get_buffer()->ptr());

        int n = 0;
        for (auto &kv_id: kv_idx) {
            auto row = kv_id.second->get_shape()[0];
            auto col = output_tensor->get_shape()[0] * output_tensor->get_shape()[1];
            std::shared_ptr<core::memory::Tensor> kv_tensor;
            if (core::memory::ModelTensorType::LLM_TENSOR_ATTN_K == output_tensor->get_tensor_type()) {
               kv_tensor = kv_cache_ctx->get_k(seq_id, layer_id, kv_id.first);
            }else if (core::memory::ModelTensorType::LLM_TENSOR_ATTN_V == output_tensor->get_tensor_type()) {
                kv_tensor = kv_cache_ctx->get_v(seq_id, layer_id, kv_id.first);
            }
            auto kv_cache_block_ptr = static_cast<T*>(kv_tensor->get_buffer()->ptr());
            T *output_block_ptr = output + n * BLOCK_ROW_SIZE * n_stride;
            for (int i = 0; i < row; i++) {
                T *output_row_ptr = output_block_ptr + i * n_stride;
                output_row_ptr = kv_cache_block_ptr + i * n_stride;
            }
            n++;
        }
    }

    template<typename T>
    void tff::kernel::GetRow<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto kv_cache_ctx = kernel::base::get_param_value<std::shared_ptr<tff::core::runtime::LLMKVCache> >(0, para_ptr);
        auto kv_idx = kernel::base::get_param_value<std::unordered_map<int, std::shared_ptr<core::memory::Tensor> > >(1, para_ptr);
        auto seq_id = kernel::base::get_param_value<int>(2, para_ptr);
        auto layer_id = kernel::base::get_param_value<int>(3, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(4, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        para_ptr->get_param_count() - 1, para_ptr);

        get_of_row<T>(seq_id, layer_id, kv_cache_ctx, kv_idx, output_tensor, stream);
    }


    template class tff::kernel::GetRow<float>;
    template class tff::kernel::GetRow<half>;
    REGISTER_OP_OBJECT(GetRow, float);

    REGISTER_OP_OBJECT(GetRow, half);
}
