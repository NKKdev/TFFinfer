//
// Created by nkk on 2025/11/2.
//

#include "LLMKVCache.h"

void tff::core::memory::LLMKVCache::build_layer_kvcache_context(const int &seq_id, const int &layer_id) {
    auto context = this->get_context(seq_id, layer_id);
    if (context) {
        tff::log::Logger::error("kv cache layer context create failed");
    }
}
