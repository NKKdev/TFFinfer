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
#include "../global/ModelGlobalVar.h"
#include "BaseDefine.h"
using namespace tff::core::global;

namespace tff::core::model {
    /**
     * @brief 字符Symbol
     */
    struct LLMSymbol {
        int32_t _prev;
        int32_t _next;
        const char *_text;
        size_t _n;
    };

    /**
     * @brief 优先队列
     */
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

    /**
     * @brief 词元元组
     */
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

    class LLMVocabulary;
    /**
     * @brief LLMTokenizer
     */
    class LLMTokenizerBase : public std::enable_shared_from_this<LLMTokenizerBase>, tff::module::ModuleObject {
    public:
        LLMTokenizerBase() = default;

        ~LLMTokenizerBase() override = default;

    public:
        virtual void init(tff::core::model::VocabPreType pre_type = TFF_VOCAB_PRE_TYPE_SMOLLM) {
        };

        virtual void tokenize(const std::string &text, std::vector<int32_t> &token,
                              const LLMVocabulary *vocabulary_ptr) {
        };

    public:
        std::vector<std::string> _regex;
    };

    /**
     * @brief BPE
     */
    class LLMTokenizerBPE : public LLMTokenizerBase {
    public:
        explicit LLMTokenizerBPE() {
        };

        ~LLMTokenizerBPE() override = default;

    public:
        void init(tff::core::model::VocabPreType pre_type = TFF_VOCAB_PRE_TYPE_SMOLLM) override;

        void tokenize(const std::string &text, std::vector<int32_t> &token,
                      const LLMVocabulary *vocabulary_ptr) override;

    protected:
        /**
         * @brief 添加元组
         * @param left 左边指针
         * @param right 右边指针
         */
        void add_tuple(int left, int right, const LLMVocabulary *vocabulary_ptr);

        /**
         * @brief 解析符号
         * @param word 字符
         */
        void parse_symbol(const std::string &word);

        /**
         * @brief 生成元组
         * @param vocabulary_ptr 词表
         */
        void generate_tuple(const LLMVocabulary *vocabulary_ptr);

        /**
         * @brief 合并元组
         * @param vocabulary_ptr 词表
         */
        void merge_tuple(const LLMVocabulary *vocabulary_ptr);

        /**
         * @brief 更新符号
         * @param symbol_vec 符号向量
         */
        void update_symbol(std::vector<LLMSymbol> &symbol_vec);

        /**
         * @brief 生成token
         * @param symbol_vec 符号向量
         * @param token_vec token向量
         * @param vocabulary_ptr 词表
         */
        void generate_token(const std::vector<LLMSymbol> &symbol_vec, const LLMVocabulary *vocabulary_ptr,
                            std::vector<int32_t> &token_vec);

    protected:
        std::vector<LLMSymbol> _symbol_vec;
        LLMVocabTupleBPE::queue _work_queue;
    };

    REGISTER_MODULE_OBJECT(LLMTokenizerBPE, LLMTokenizerBase, "TOKENIZER",
                           get_tokenizer_name(tff::core::model::VocabType::TFF_VOCAB_TYPE_BPE).data());
}


#endif //TFFINFER_LLMTOKENIZERBASE_H
