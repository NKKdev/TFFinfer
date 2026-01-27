//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "mem/LLMKVCache.h"

namespace tff::kernel {
    template<typename T>
    void get_of_row(const int seq_id, const int layer_id,
                    std::shared_ptr<tff::core::memory::LLMKVCache> &kv_cache_ctx,
                    std::unordered_map<int, std::shared_ptr<core::memory::Tensor> > &kv_idx,
                    std::shared_ptr<core::memory::Tensor> &output_tensor,
                    std::shared_ptr<core::device::DeviceStream> &stream,
                    std::shared_ptr<core::device::DeviceEvent> &event,
                    std::vector<std::shared_ptr<core::device::DeviceEvent> > &event_list) {
        constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;
        const int n_stride = output_tensor->get_shape()[0];
        T *output = static_cast<T *>(output_tensor->get_buffer()->ptr());
        for (auto &wait_event: event_list) {
            stream->wait_event(wait_event->get_native_event());
        }
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
        event->record(stream);
    }

    template<typename T>
    void tff::kernel::GetRow<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), GetRow<T>::get_op_name().c_str());
        auto kv_cache_ctx = get_param_value<std::shared_ptr<tff::core::memory::LLMKVCache> >(1, para_ptr);
        auto kv_idx = get_param_value<std::unordered_map<int, std::shared_ptr<core::memory::Tensor> > >(2, para_ptr);
        auto seq_id = get_param_value<int>(3, para_ptr);
        auto layer_id = get_param_value<int>(4, para_ptr);
        auto output_tensor = get_param_value<std::shared_ptr<core::memory::Tensor> >(5, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(6, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream> >(7, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent> >(8, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent> > >(9, para_ptr);
        if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr || kv_cache_ctx == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }
        get_of_row<T>(seq_id, layer_id, kv_cache_ctx, kv_idx, output_tensor, stream, event, event_list);
    }

    template<typename T>
    std::string tff::kernel::GetRow<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_GET_ROWS);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;

        return name;
    }

    template class tff::kernel::GetRow<float>;
    template class tff::kernel::GetRow<half>;
    REGISTER_OP_OBJECT(GetRow, float);

    REGISTER_OP_OBJECT(GetRow, half);
}
