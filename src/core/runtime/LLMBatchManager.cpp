//
// Created by nkk on 2025/11/2.
//

#include "LLMBatchManager.h"
namespace tff::core::runtime {
    bool tff::core::runtime::LLMBatchManager::init(const std::map<int, std::string> &seq_prompt,
        std::shared_ptr<tff::core::model::LLMLLaMaVocabulary> &vocabulary_ptr, bool output_all) {
        _vocabulary = vocabulary_ptr;
        _main_batch = std::make_shared<LLMBatch>();
        for (const auto& [sid, prompt] : seq_prompt) {
            std::shared_ptr<LLMSeq> llm_seq = std::make_shared<LLMSeq>(sid, _vocabulary->get_eos_tokens(),
                this->_vocabulary->get_eog_tokens());
            this->_sequence.insert(std::make_pair(sid, std::move(llm_seq)));
            auto iter = this->_sequence.find(sid);
            std::vector<int32_t> tokens;
            _vocabulary->tokenize(prompt, tokens);
            for (size_t i = 0; i < tokens.size(); ++i) {
                const auto token = tokens[i];
                bool need_logits = output_all || (i == tokens.size() - 1);
                _main_batch->_n_tokens += i;
                _main_batch->_tokens.push_back(token);
                iter->second->append_token(token);

            }
        }

        return true;
    }
}
