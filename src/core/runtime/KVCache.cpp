//
// Created by nkk on 2025/11/2.
//

#include "KVCache.h"

namespace tff::core::runtime {
    void LLMKVCache::build_layer_kvcache_context(const int &seq_id, const int &layer_id) {
        auto context = this->get_context(seq_id, layer_id);
        if (context == nullptr) {
            tff::log::Logger::error("kv cache layer context create failed");
        }
    }

    //
    std::unordered_map<int, std::shared_ptr<memory::Tensor> > LLMKVCache::set_k(
        int seq_id, int layer_id,
        const std::shared_ptr<core::memory::Tensor> &cur_k,
        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > &device) {
        LayerKVContext *ctx = get_context(seq_id, layer_id);
        auto token_num = cur_k->get_shape()[2];
        std::unordered_map<int, std::vector<int64_t> > k_idx;
        for (int i = 0; i < token_num; i++) {
            if (!ctx->append_token()) {
                return std::unordered_map<int, std::shared_ptr<core::memory::Tensor> >(); // 内存不足
            }
            int token_pos = ctx->_num_tokens - 1;

            auto index_pair = ctx->get_location(token_pos);
            const int page_idx = index_pair.first;
            const int offset = index_pair.second;
            auto iter = k_idx.find(page_idx);
            if (iter == k_idx.end()) {
                k_idx.insert(std::make_pair(page_idx, std::vector<int64_t>({offset})));
            } else {
                iter->second.push_back(offset);
            }
        }
        std::unordered_map<int, std::shared_ptr<core::memory::Tensor> > kv_idx_device;
        auto device_iter = device.begin();
        for (auto &idx: k_idx) {
            auto tensor = std::make_shared<core::memory::Tensor>(memory::DataType::TFF_DATA_TYPE_I64,
                memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                 std::array<int64_t, MAX_TENSOR_DIM>{
                                                                     static_cast<int64_t>(idx.second.size()), 1, 1, 1
                                                                 }, false,
                                                                 device_iter->second->get_device_buffer_allocator(
                                                                     device_iter->first));
            tensor->get_allocator()->memcopy(static_cast<void*>(idx.second.data()),
                static_cast<void*>(tensor->get_buffer()->ptr()), idx.second.size() * sizeof(int64_t),
                memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE);
            auto iter = kv_idx_device.find(idx.first);
            if (iter == kv_idx_device.end()) {
                kv_idx_device.insert(std::make_pair(idx.first, tensor));
            }
        }
        return kv_idx_device;
    }
}
