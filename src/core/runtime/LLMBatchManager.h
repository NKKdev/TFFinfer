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
    class LLMSeq :public std::enable_shared_from_this<LLMSeq>{
    public:
        LLMSeq() = default;

        LLMSeq(const int32_t id, const std::vector<int32_t> &eos_ids, const std::vector<int32_t> &eog_ids) : _seq_id(id), _is_finished(false),
                                                                               _eos_token_ids(eos_ids),
                                                                               _eog_token_ids(eog_ids),
        _parent_seq_id(-1),_shared_prefix_len(0), _is_prefilled(false){
        }

        ~LLMSeq() = default;

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

        int32_t _parent_seq_id;           // 父 sequence ID，-1 表示无父
        uint32_t _shared_prefix_len;      // 与父共享多少个 token
        bool _is_prefilled;               // 是否已初始化（避免重复处理）
    };

    class LLMBatch :public std::enable_shared_from_this<LLMBatch>{
    public:
        LLMBatch() = default;

        ~LLMBatch() = default;

    public:
        uint32_t _n_tokens;
        std::vector<uint32_t> _tokens;
        std::vector<float> _embd;
        std::vector<int32_t> _pos;
        std::vector<int32_t> _token_seq_ids;
        std::vector<int8_t> _logits;
    };

    class LLMBatchManager final :public tff::module::ModuleObject{
    public:
        LLMBatchManager() = default;

        ~LLMBatchManager() override = default;

    public:
        bool init(const std::unordered_map<int, std::string> &seq_prompt,
                  const std::shared_ptr<tff::core::model::LLMLLaMaVocabulary> &vocabulary_ptr,
                  bool output_all = false);

        // 分割策略 1：简单打包（填满为止）
        void split_simple();

        // 分割策略 2：等长 sequence 分组
        void split_equal();

        // 分割策略 3：每个 ubatch 一个 sequence（流式生成）
        void split_seq();

        // 获取所有 sequence 的当前状态（用于 KV Cache 管理）
        [[nodiscard]] const std::map<uint32_t, std::shared_ptr<LLMSeq> > &get_sequence() const {
            return this->_sequence;
        }

    private:
        // 辅助函数：根据策略分组 token 索引
        std::vector<std::vector<int> > group_tokens_by_strategy(const std::string &strategy);

        // 构建一个 llama_batch 从 token indices
        std::shared_ptr<LLMBatch> build_sub_batch(const std::vector<int> &token_indices);

    public:
        uint32_t _n_stream{};
        int32_t _max_batch_size{};
        int32_t _max_seq_size{};
        std::shared_ptr<LLMBatch> _main_batch;
        std::vector<std::shared_ptr<LLMBatch> > _ubatches;
        std::shared_ptr<tff::core::model::LLMLLaMaVocabulary> _vocabulary;
        std::map<uint32_t, std::shared_ptr<LLMSeq> > _sequence;
    };
}

#endif //TFFINFER_LLMBATCHMANAGER_H
