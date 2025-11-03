//
// Created by nkk on 2025/11/2.
//

#include "LLMBatchManager.h"
#include "Logger.h"

namespace tff::core::runtime {
    bool tff::core::runtime::LLMBatchManager::init(const std::map<int, std::string> &seq_prompt,
                                                   const std::shared_ptr<tff::core::model::LLMLLaMaVocabulary> &
                                                   vocabulary_ptr, const bool output_all) {
if (!vocabulary_ptr) {
        tff::log::Logger::error("LLMBatchManager::init: vocabulary is null");
        return false;
    }

    this->_vocabulary = vocabulary_ptr;
    this->_sequence.clear();
    this->_main_batch = std::make_shared<LLMBatch>();
    this->_max_seq_size = 0;
    this->_max_batch_size = 0;
    this->_ubatches.clear();

    std::vector<uint32_t> batch_tokens;
    std::vector<int32_t> batch_pos;
    std::vector<int8_t> batch_logits;
    // Step 1: 所有 prompt 分词并排序（短优先，便于前缀匹配）
    std::vector<std::pair<int32_t, std::vector<int32_t>>> prompts;
    for (const auto& [sid, prompt] : seq_prompt) {
        auto seq_id = static_cast<int32_t>(sid);
        auto eos_ids = this->_vocabulary->get_eos_tokens();
        auto eog_ids = this->_vocabulary->get_eog_tokens();
        auto llm_seq = std::make_shared<LLMSeq>(seq_id, eos_ids, eog_ids);
        this->_sequence[seq_id] = llm_seq;

        std::vector<int32_t> tokens;
        this->_vocabulary->tokenize(prompt, tokens);
        if (tokens.empty()) {
            tff::log::Logger::warning("Empty tokenized result for seq_id=%d", seq_id);
            continue;
        }

        prompts.emplace_back(seq_id, std::move(tokens));
    }

    // 按长度升序排列，方便找前缀
    std::sort(prompts.begin(), prompts.end(),
              [](const auto& a, const auto& b) {
                  return a.second.size() < b.second.size();
              });

    // Step 2: 构建共享关系
    for (size_t i = 0; i < prompts.size(); ++i) {
        const int32_t seq_id = prompts[i].first;
        const std::vector<int32_t>& tokens = prompts[i].second;
        auto llm_seq = this->_sequence[seq_id];

        // 查找最佳父 sequence（最长前缀匹配）
        int32_t best_parent_id = -1;
        uint32_t max_shared_len = 0;

        for (size_t j = 0; j < i; ++j) {
            const int32_t parent_id = prompts[j].first;
            const auto& parent_tokens = prompts[j].second;
            const auto parent_seq = _sequence[parent_id];
            if (!parent_seq || parent_seq->_is_finished) continue;

            size_t shared_len = 0;
            const size_t min_len = std::min(tokens.size(), parent_tokens.size());
            while (shared_len < min_len && tokens[shared_len] == parent_tokens[shared_len]) {
                ++shared_len;
            }

            if (shared_len > max_shared_len) {
                max_shared_len = static_cast<uint32_t>(shared_len);
                best_parent_id = parent_id;
            }
        }

        // 设置共享信息
        if (best_parent_id != -1 && max_shared_len > 0) {
            llm_seq->_parent_seq_id = best_parent_id;
            llm_seq->_shared_prefix_len = max_shared_len;
            tff::log::Logger::info("seq_id=%d shares %u tokens with parent=%d",
                                    seq_id, max_shared_len, best_parent_id);
        }

        // 写入 tokens 到 sequence
        for (int32_t token : tokens) {
            llm_seq->append_token(token);
            if (llm_seq->_is_finished) break;
        }
        llm_seq->_is_prefilled = true;

        // Step 3: 将非共享部分加入 batch
        const uint32_t start_pos = llm_seq->_shared_prefix_len;
        for (size_t pos_in_seq = start_pos; pos_in_seq < tokens.size(); ++pos_in_seq) {
            const int32_t token = tokens[pos_in_seq];

            if (token < 0 || token >= static_cast<int32_t>(_vocabulary->_id_to_token.size())) {
                tff::log::Logger::error("Invalid token[%zu]=%d in seq_id=%d", pos_in_seq, token, seq_id);
                return false;
            }

            batch_tokens.push_back(static_cast<uint32_t>(token));
            batch_pos.push_back(static_cast<int32_t>(pos_in_seq));
            batch_logits.push_back((output_all || pos_in_seq == tokens.size() - 1) ? 1 : 0);
            this->_main_batch->_token_seq_ids.emplace_back(static_cast<int32_t>(seq_id));
        }
    }

    // Step 4: 填充主 batch
    this->_main_batch->_n_tokens = static_cast<uint32_t>(batch_tokens.size());
    this->_main_batch->_tokens = std::move(batch_tokens);
    this->_main_batch->_pos = std::move(batch_pos);
    this->_main_batch->_logits = std::move(batch_logits);

    this->_max_seq_size = static_cast<int32_t>(_sequence.size());

    tff::log::Logger::info("LLMBatchManager::init: %d sequences, %u non-shared tokens in batch",
                           this->_max_seq_size, this->_main_batch->_n_tokens);

    return true;
    }

    void LLMBatchManager::split_simple() {
        const uint32_t max_tokens_per_ubatch = this->_max_batch_size > 0 ? this->_max_batch_size : MAX_BATCH_SIZE; // 默认值
        const uint32_t total_tokens = _main_batch->_tokens.size();

        for (uint32_t i = 0; i < total_tokens; ) {
            std::vector<int> indices;
            uint32_t end = std::min(i + max_tokens_per_ubatch, total_tokens);
            for (uint32_t j = i; j < end; ++j) {
                indices.push_back(static_cast<int>(j));
            }
            this->_ubatches.push_back(build_sub_batch(indices));
            i = end;
        }

        tff::log::Logger::info("LLMBatchManager::split_simple: split into %zu ubatches (max_tokens=%u)",
                               this->_ubatches.size(), max_tokens_per_ubatch);

    }
    static bool compare_seq_size(const std::vector<int> &seq1, const std::vector<int> &seq2) {
        return seq1.size() < seq2.size();
    }
    void LLMBatchManager::split_equal() {
        auto &ubatches = this->_ubatches;

        //
        std::unordered_map<int32_t, std::vector<int>> temp_map;
        temp_map.reserve(_max_seq_size); // 预分配，避免 rehash

        const auto& token_seq_ids = _main_batch->_token_seq_ids;
        for (size_t i = 0; i < token_seq_ids.size(); ++i) {
            int32_t seq_id = token_seq_ids[i];
            auto& indices = temp_map[seq_id];
            indices.push_back(static_cast<int>(i));
        }

        //从较短的seq开始查找共享前缀;
        std::vector<std::pair<int32_t, std::vector<int>>> sorted_seqs;
        sorted_seqs.reserve(temp_map.size());

        for (auto& [sid, indices] : temp_map) {
            sorted_seqs.emplace_back(sid, std::move(indices));
        }

        std::sort(sorted_seqs.begin(), sorted_seqs.end(),
            [](const auto& a, const auto& b) {
                return a.second.size() < b.second.size();
            });

        //按seq分组;
        const int max_seqs_per_ubatch = (this->_max_batch_size > 0) ? this->_max_batch_size : MAX_BATCH_SIZE;
        for (size_t i = 0; i < sorted_seqs.size(); ) {
            std::vector<int> indices;
            int count = 0;
            while (i < sorted_seqs.size() && count < max_seqs_per_ubatch) {
                auto& tok_indices = sorted_seqs[i].second;
                indices.insert(indices.end(), tok_indices.begin(), tok_indices.end());
                ++i; ++count;
            }
            if (!indices.empty()) {
                std::sort(indices.begin(), indices.end()); // 保持 token 顺序
                ubatches.push_back(build_sub_batch(indices));
            }
        }
    }

    void LLMBatchManager::split_seq() {
        auto &ubatches = this->_ubatches;
        // 按 seq_id 收集 token indices
        std::map<int32_t, std::vector<int>> seq_to_token_indices;
        for (size_t i = 0; i < _main_batch->_token_seq_ids.size(); ++i) {
            int32_t seq_id = _main_batch->_token_seq_ids[i];
            seq_to_token_indices[seq_id].push_back(static_cast<int>(i));
        }

        // 每个 sequence 单独成一个 batch
        for (const auto& [seq_id, indices] : seq_to_token_indices) {
            if (!indices.empty()) {
                auto ubatch = build_sub_batch(indices);
                ubatches.push_back(ubatch);
                tff::log::Logger::info("LLMBatchManager::split_seq: seq_id=%d -> %u tokens", seq_id, ubatch->_n_tokens);
            }
        }

        tff::log::Logger::info("LLMBatchManager::split_seq: created %zu ubatches (one per sequence)", ubatches.size());
    }
    //
    std::shared_ptr<LLMBatch> LLMBatchManager::build_sub_batch(const std::vector<int>& token_indices) {
        auto sub_batch = std::make_shared<LLMBatch>();
        sub_batch->_n_tokens = static_cast<uint32_t>(token_indices.size() - 1);

        for (const int idx : token_indices) {
            sub_batch->_tokens.push_back(_main_batch->_tokens[idx]);
            sub_batch->_pos.push_back(_main_batch->_pos[idx]);
            sub_batch->_logits.push_back(_main_batch->_logits[idx]);
            sub_batch->_token_seq_ids.push_back(_main_batch->_token_seq_ids[idx]);
        }

        return sub_batch;
    }
}
