//
// Created by nkk on 2025/11/2.
//

#ifndef TFFINFER_LLMBATCHMANAGER_H
#define TFFINFER_LLMBATCHMANAGER_H
#include <cstdint>
#include <vector>
#include <set>
#include <string>
#include <map>
#include <memory>
#include "model/LLMVocabulary.h"

namespace tff::core::runtime {
    class LLMSeq {
    public:
        LLMSeq();

        LLMSeq(const int32_t id, const std::vector<int32_t> &eos_ids, const std::vector<int32_t> &eog_ids) : _seq_id(id), _is_finished(false),
                                                                               _eos_token_ids(eos_ids),
                                                                               _eog_token_ids(eog_ids) {
        }

        ~LLMSeq();

    public:
        inline void append_token(const int32_t token) {
            if (!_is_finished) {
                _tokens.push_back(token);
                this->_current_pos = this->_tokens.size() - 1;
                auto iter = std::ranges::find(_eos_token_ids, token);
                if (iter != _eos_token_ids.end()) {
                    _is_prompt_done = true;
                }
                auto iter_eog = std::ranges::find(_eog_token_ids, token);
                if (iter_eog != _eog_token_ids.end()) {
                    _is_finished = true;
                }
            }
        }

        [[nodiscard]] inline bool can_generate() const {
            return !_is_finished;
        }

    public:
        int32_t _seq_id; // 序列的唯一ID
        std::vector<int32_t> _tokens; // 已生成的token序列 (包括prompt)
        bool _is_finished; // 是否已完成 (遇到EOS)
        std::vector<int32_t> _eos_token_ids; // EOS token的ID
        std::vector<int32_t> _eog_token_ids;
        int _current_pos = 0;
        bool _is_prompt_done = false;
    };

    class LLMBatch {
    public:
        LLMBatch();

        ~LLMBatch();

    public:
        uint32_t _n_tokens;
        std::vector<int32_t> _tokens;
        std::vector<float> _embd;
        std::vector<int32_t> _pos;
        std::vector<int32_t> _n_seq_id;
        std::vector<std::vector<int32_t> > _seq_id_belong_to;
        std::vector<int8_t> _logits;
        std::set<int32_t> _seq_id_unq;
    };

    class LLMBatchManager {
    public:
        LLMBatchManager();

        ~LLMBatchManager();

    public:
        bool init(const std::map<int, std::string> &seq_prompt,
                  std::shared_ptr<tff::core::model::LLMLLaMaVocabulary> &vocabulary_ptr,
                  bool output_all = false);

        // 分割策略 1：简单打包（填满为止）
        std::vector<std::shared_ptr<LLMBatch> > split_simple();

        // 分割策略 2：等长 sequence 分组
        std::vector<std::shared_ptr<LLMBatch> > split_equal();

        // 分割策略 3：每个 ubatch 一个 sequence（流式生成）
        std::vector<std::shared_ptr<LLMBatch> > split_seq();

        // 获取所有 sequence 的当前状态（用于 KV Cache 管理）
        const std::map<uint32_t, std::shared_ptr<LLMSeq> > &get_sequence() const;

    private:
        // 辅助函数：根据策略分组 token 索引
        std::vector<std::vector<int> > group_tokens_by_strategy(const std::string &strategy);

        // 构建一个 llama_batch 从 token indices
        std::shared_ptr<LLMBatch> build_sub_batch(const std::vector<int> &token_indices);

    public:
        int32_t _n_batch;
        int32_t _n_seq;
        std::shared_ptr<LLMBatch> _main_batch;
        std::vector<std::shared_ptr<LLMBatch> > _ubatches;
        std::shared_ptr<tff::core::model::LLMLLaMaVocabulary> _vocabulary;
        std::map<uint32_t, std::shared_ptr<LLMSeq> > _sequence;
    };
}

#endif //TFFINFER_LLMBATCHMANAGER_H
