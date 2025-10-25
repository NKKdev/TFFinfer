//
// Created by nkk on 2025/10/24.
//

#include "LLMTokenizerBase.h"

#include "LLMVocabulary.h"
#include "unicode.h"

namespace tff::core::model {
    void tff::core::model::LLMTokenizerBPE::tokenize(const std::string &text, std::vector<int32_t> &token,
        const LLMLLaMaVocabulary *vocabulary_ptr) {
        const auto byte_vec = unicode_regex_split(text, this->_regex);
        std::vector<LLMSymbol> symbol_vec_tmp;
        for (const auto & word : byte_vec) {
            this->parse_symbol(word);
            this->generate_tuple(vocabulary_ptr);
            this->merge_tuple(vocabulary_ptr);
            this->update_symbol(symbol_vec_tmp);
        }
        this->generate_token(symbol_vec_tmp, vocabulary_ptr, token);
    }
    //
    void tff::core::model::LLMTokenizerBPE::add_tuple(int left, int right,
        const LLMLLaMaVocabulary *vocabulary_ptr) {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token  = std::string(_symbol_vec[left]._text,  _symbol_vec[left]._n);
        std::string right_token = std::string(_symbol_vec[right]._text, _symbol_vec[right]._n);

        int rank_found = vocabulary_ptr->get_rank(left_token, right_token);
        if (rank_found < 0) {
            return;
        }

        LLMVocabTupleBPE tuple_bpe;

        tuple_bpe._left  = left;
        tuple_bpe._right = right;
        tuple_bpe._text  = left_token + right_token;
        tuple_bpe._size  = left_token.size() + right_token.size();
        tuple_bpe._rank  = rank_found;

        _work_queue.push(tuple_bpe);
    }

    void LLMTokenizerBPE::parse_symbol(const std::string &word) {
        _symbol_vec.clear();
        int index = 0;
        size_t offset = 0;
        while (offset < word.size()) {
            LLMSymbol sym{};
            size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
            sym._text = word.c_str() + offset;
            sym._n = char_len;
            offset += sym._n;
            sym._prev = index - 1;
            sym._next = offset == word.size() ? -1 : index + 1;
            index++;
            _symbol_vec.emplace_back(sym);
        }
    }

    void LLMTokenizerBPE::generate_tuple(const LLMLLaMaVocabulary *vocabulary_ptr) {
        for (size_t i = 1; i < _symbol_vec.size(); ++i) {
            this->add_tuple(i - 1, i, vocabulary_ptr);
        }
    }

    void LLMTokenizerBPE::merge_tuple(const LLMLLaMaVocabulary *vocabulary_ptr) {
        while (!_work_queue.empty()) {
            auto tuple = _work_queue.pop_move();
            auto & left_symbol = _symbol_vec[tuple._left];
            auto & right_symbol = _symbol_vec[tuple._right];

            if (left_symbol._n == 0 || right_symbol._n == 0) {
                continue;
            }
            std::string left_token = std::string(left_symbol._text, left_symbol._n);
            std::string right_token = std::string(right_symbol._text, right_symbol._n);
            if (left_token + right_token != tuple._text) {
                continue;  // Skip this bigram if it's outdated
            }

            // merge the right sym into the left one
            left_symbol._n += right_symbol._n;
            right_symbol._n = 0;

            // remove the right sym from the chain
            left_symbol._next = right_symbol._next;
            if (right_symbol._next >= 0) {
                _symbol_vec[right_symbol._next]._prev = tuple._left;
            }

            this->add_tuple(left_symbol._prev, tuple._left, vocabulary_ptr);  // left side of current symbol
            this->add_tuple(tuple._left, left_symbol._next, vocabulary_ptr);  // right side of current symbol
        }
    }

    void LLMTokenizerBPE::update_symbol(std::vector<LLMSymbol> &symbol_vec) {
        int32_t prev_index = -1;
        for (auto & sym : this->_symbol_vec) {
            if (sym._n > 0) {
                sym._prev = prev_index;
                sym._next = -1;
                if (prev_index != -1) {
                    symbol_vec[prev_index]._next = symbol_vec.size();
                }
                symbol_vec.emplace_back(sym);
                prev_index = symbol_vec.size() - 1;
            }
        }
    }
    //
    void LLMTokenizerBPE::generate_token(const std::vector<LLMSymbol> &symbol_vec,
        const LLMLLaMaVocabulary *vocabulary_ptr,
        std::vector<int32_t> &token_vec) {
        if (!symbol_vec.empty()) {
            for (int i = 0; i != -1; i = symbol_vec[i]._next) {
                auto & symbol = symbol_vec[i];
                if (symbol._n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol._text, symbol._n);
                const auto token = vocabulary_ptr->text_to_token(str);

                if (token == LLAMA_TOKEN_NULL) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocabulary_ptr->text_to_token(byte_str);
                        if (token_multibyte != -1) {
                            token_vec.push_back(token_multibyte);
                        }
                    }
                } else {
                    token_vec.push_back(token);
                }
            }
        }
    }
}
