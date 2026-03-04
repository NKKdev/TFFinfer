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
    bool LLMKVCache::set_kv(
    const int seq_id, const int layer_id,
               const int &token_num) {
        LayerKVContext *ctx = get_context(seq_id, layer_id);
        if (ctx->get_token_count() == token_num) {
            return true;
        }
        for (int i = 0; i < token_num; i++) {
            if (!ctx->append_token()) {
                return false; // 内存不足
            }
        }
        return true;
    }
}
