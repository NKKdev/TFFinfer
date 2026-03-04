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
    /**
     * @brief 大模型文本序列
     */
    class LLMSeq : public std::enable_shared_from_this<LLMSeq> {
    public:
        LLMSeq() = default;

        LLMSeq(const int32_t id, const std::vector<int32_t> &eos_ids,
               const std::vector<int32_t> &eog_ids) : _seq_id(id), _is_finished(false),
                                                      _eos_token_ids(eos_ids),
                                                      _eog_token_ids(eog_ids),
                                                      _parent_seq_id(-1), _shared_prefix_len(0), _is_prefilled(false) {
        }

        ~LLMSeq() = default;

    public:
        /**
         * @brief 添加一个 token 到序列中
         * @param token
         */
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

        /**
         * @brief 是否可以生成
         * @return
         */
        [[nodiscard]] inline bool can_generate() const {
            return !_is_finished;
        }

    public:
        int32_t _seq_id;
        std::vector<int32_t> _tokens;
        bool _is_finished;
        std::vector<int32_t> _eos_token_ids;
        std::vector<int32_t> _eog_token_ids;
        int _current_pos = 0;
        bool _is_prompt_done = false;

        int32_t _parent_seq_id;
        uint32_t _shared_prefix_len;
        bool _is_prefilled;
    };

    /**
     * @brief 大模型 文本batch
     */
    class LLMBatch : public std::enable_shared_from_this<LLMBatch> {
    public:
        LLMBatch() = default;

        ~LLMBatch() = default;

        LLMBatch &operator=(const LLMBatch &other) {
            this->_n_tokens = other._n_tokens;
            this->_logits = other._logits;
            this->_tokens = other._tokens;
            this->_embd = other._embd;
            this->_pos = other._pos;
            this->_token_seq_ids = other._token_seq_ids;
        };

    public:
        uint32_t _n_tokens;
        std::vector<uint32_t> _tokens;
        std::vector<float> _embd;
        std::vector<int32_t> _pos;
        std::vector<int32_t> _token_seq_ids;
        std::vector<int8_t> _logits;
    };

    /**
     * @brief 大模型 batch 管理器
     */
    class LLMBatchManager final : public tff::module::ModuleObject {
    public:
        enum BatchSplitType {
            TTF_BATCH_SPLIT_SIMPLE = 0,
            TTF_BATCH_SPLIT_EQUAL = 1,
            TTF_BATCH_SPLIT_SEQ = 2,
        };

    public:
        LLMBatchManager() = default;

        ~LLMBatchManager() override = default;

    public:
        /**
         * @brief 初始化
         * @param seq_prompt 模型输入
         * @param vocabulary_ptr 词表
         * @param output_all 是否输出所有
         * @param batch_split_type 分割策略
         * @return
         */
        bool init(const std::unordered_map<int, std::string> &seq_prompt,
                  const std::shared_ptr<tff::core::model::LLMVocabulary> &vocabulary_ptr,
                  bool output_all = false, BatchSplitType batch_split_type = TTF_BATCH_SPLIT_SEQ);

        /**
         * @brief 分割策略 1：简单分组
         */
        void split_simple();

        /**
         * @brief 分割策略 2：每个 ubatch 一个 batch（批量生成, 长度相等）
         */
        void split_equal();

        /**
         * @brief 分割策略 3：每个 sequence 一个 batch
         */
        void split_seq();

        /**
         * @brief 获取序列
         * @return
         */
        [[nodiscard]] const std::map<uint32_t, std::shared_ptr<LLMSeq> > &get_sequence() const {
            return this->_sequence;
        }

    private:
        /**
         * @brief 根据策略分组
         * @param strategy 分组策略
         * @return
         */
        std::vector<std::vector<int> > group_tokens_by_strategy(const std::string &strategy);

        /**
         * @brief 构建子 batch
         * @param token_indices
         * @return
         */
        std::shared_ptr<LLMBatch> build_sub_batch(const std::vector<int> &token_indices);

    public:
        uint32_t _n_stream{};
        int32_t _max_batch_size{};
        int32_t _max_seq_size{};
        std::shared_ptr<LLMBatch> _main_batch;
        std::vector<std::shared_ptr<LLMBatch> > _ubatches;
        std::shared_ptr<tff::core::model::LLMVocabulary> _vocabulary;
        std::map<uint32_t, std::shared_ptr<LLMSeq> > _sequence;
    };
}

#endif //TFFINFER_LLMBATCHMANAGER_H
