//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_LLMTOKENIZERBASE_H
#define TFFINFER_LLMTOKENIZERBASE_H
#include <memory>
#include <vector>
#include <queue>
#include "ModuleFactory.h"
#include "ModuleObject.h"
#include "ModelGlobalVar.h"
#include "BaseDefine.h"
namespace tff::core::model {
    struct LLMSymbol {
        int32_t _prev;
        int32_t _next;
        const char *_text;
        size_t _n;
    };

    template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename
        Container::value_type> >
    class LLMPriorityQueue : public std::priority_queue<T, Container, Compare> {
    public:
        using std::priority_queue<T, Container, Compare>::priority_queue;

        T pop_move() {
            T item = std::move(this->c.front());
            std::pop_heap(this->c.begin(), this->c.end(), this->comp);
            this->c.pop_back();
            return item;
        }

        void pop() = delete;
    };

    class LLMVocabTupleBPE {
    public:
        LLMVocabTupleBPE() = default;

        ~LLMVocabTupleBPE() = default;

    public:
        struct comparator {
            bool operator()(const LLMVocabTupleBPE &l, const LLMVocabTupleBPE &r) const {
                return l._rank > r._rank || (l._rank == r._rank && l._left > r._left);
            }
        };

        using queue_storage = std::vector<LLMVocabTupleBPE>;
        using queue = LLMPriorityQueue<LLMVocabTupleBPE, queue_storage, comparator>;
        int32_t _left{};
        int32_t _right{};
        std::string _text;
        int _rank{};
        size_t _size{};
    };
    class LLMLLaMaVocabulary;
    class LLMTokenizerBase : public std::enable_shared_from_this<LLMTokenizerBase>,tff::module::ModuleObject {
    public:
        LLMTokenizerBase() = default;

        ~LLMTokenizerBase() override = default;

    public:
        virtual void tokenize(const std::string &text, std::vector<int32_t> &token, const LLMLLaMaVocabulary *vocabulary_ptr){};

    public:
        std::vector<std::string> _regex;
    };

    class LLMTokenizerBPE : public LLMTokenizerBase {
    public:
        explicit LLMTokenizerBPE(tff::core::model::VocabPreType pre_type = TFF_VOCAB_PRE_TYPE_SMOLLM) {
            LLM_SPECIAL_TOKENS[LLM_KV_TOKENIZER_BOS_ID]  = 11;
            LLM_SPECIAL_TOKENS[LLM_KV_TOKENIZER_EOS_ID] = 11;
            LLM_SPECIAL_TOKENS[LLM_KV_TOKENIZER_UNK_ID] = -1;
            LLM_SPECIAL_TOKENS[LLM_KV_TOKENIZER_SEP_ID] = -1;
            LLM_SPECIAL_TOKENS[LLM_KV_TOKENIZER_PAD_ID] = -1;
            LLM_SPECIAL_TOKENS[LLM_KV_TOKENIZER_MASK_ID] = -1;
            if (pre_type == TFF_VOCAB_PRE_TYPE_SMOLLM) {
                _regex = {
                    "\\p{N}",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
            }
        };

        ~LLMTokenizerBPE() override= default;
    public:
        void tokenize(const std::string &text, std::vector<int32_t> &token, const LLMLLaMaVocabulary *vocabulary_ptr) override;
    protected:
        //
        void add_tuple(int left, int right, const LLMLLaMaVocabulary *vocabulary_ptr);
        //
        void parse_symbol(const std::string &word);
        //
        void generate_tuple(const LLMLLaMaVocabulary *vocabulary_ptr);
        //
        void merge_tuple(const LLMLLaMaVocabulary *vocabulary_ptr);
        //
        void update_symbol(std::vector<LLMSymbol> &symbol_vec);
        //
        void generate_token(const std::vector<LLMSymbol> &symbol_vec, const LLMLLaMaVocabulary *vocabulary_ptr, std::vector<int32_t> &token_vec);
    protected:
        std::vector<LLMSymbol> _symbol_vec;
        LLMVocabTupleBPE::queue _work_queue;
    };

    REGISTER_MODULE_OBJECT(LLMTokenizerBPE, LLMTokenizerBase,"TOKENIZER",
                           get_tokenizer_name(tff::core::model::VocabType::TFF_VOCAB_TYPE_BPE).data());
}


#endif //TFFINFER_LLMTOKENIZERBASE_H
