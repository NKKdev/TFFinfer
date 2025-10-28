//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_LLMVOCABULARY_H
#define TFFINFER_LLMVOCABULARY_H
#include "BaseDefine.h"
#include <unordered_map>
#include "ModuleFactory.h"
#include "ModuleObject.h"
#include "mem/Tensor.h"
#include "model/base/ModelLoaderBase.h"
#include "LLMTokenizerBase.h"
#include "util.h"
#include <set>
#include <queue>

namespace tff::core::model {
    class LLMLLaMaVocabulary : public tff::module::ModuleObject {
    public:
        LLMLLaMaVocabulary() : _n_token_types(0), _vaocabulary_type(tff::core::model::VocabType::TFF_VOCAB_TYPE_BPE),
                               _add_bos(false), _add_space_prefix(false), _add_eos(false), _add_sep(false),
                               _ignore_merges(false), _clean_spaces(false), _remove_extra_whitespaces(false),
                               _escape_whitespaces(false), _treat_whitespace_as_suffix(false) {
        };

        ~LLMLLaMaVocabulary() override = default;

    public:
        bool load_vocabulary(const std::shared_ptr<tff::core::model::ModelLoaderBase> &_model_loader);

        int32_t get_rank(const std::string &token_left, const std::string &token_right) const;
        //
        inline int32_t text_to_token(const std::string & text) const {
            auto it = this->_token_to_id.find(text);
            if (it != this->_token_to_id.end()) {
                return (*it).second;
            }
            return -1;
        }

    protected:
        bool load_bpe();

        //
        bool load_token_data();

        //
        void tokenize(const std::string &raw_text,
                      std::vector<int32_t> &token_vec,
                      bool add_special, bool parse_special) const;
        //
        void process_special_tokens();
        //
        void process_eog_tokens();
        //
        void process_user_defined_tokens();

        std::string token_to_string(const int32_t &token, bool special);

        //
        int32_t token_to_string(int32_t token, char *buf, int32_t length, int32_t lstrip, bool special = true);

    protected:
        //
        std::string _tokenizer_model;
        std::string _tokenizer_pre;

        // tokenizer flags
        bool _add_space_prefix;
        bool _add_bos;
        bool _add_eos;
        bool _add_sep;
        bool _ignore_merges;
        bool _clean_spaces;
        bool _remove_extra_whitespaces;
        bool _escape_whitespaces;
        bool _treat_whitespace_as_suffix;
        //
        uint32_t _n_token_types;
        //
        tff::core::model::VocabType _vaocabulary_type;
        tff::core::model::VocabPreType _vacabulary_pre_type;
        //
        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;
        //
        std::shared_ptr<tff::core::model::LLMTokenizerBase> _tokenizer;
        //
        std::unordered_map<std::string, int32_t> _token_to_id;
        std::vector<tff::core::model::TokenData> _id_to_token;
        //
        std::unordered_map<std::pair<std::string, std::string>, int, tff::utils::pair_hash> _bpe_ranks;

        //
        int32_t _linefeed_id;
        std::set<uint32_t> _special_eog_ids;
        //
        std::vector<uint32_t> _cache_special_tokens;
        std::vector<std::string> _cache_token_to_piece;

    private:

    };

    REGISTER_MODULE_OBJECT(LLMLLaMaVocabulary, tff::module::ModuleObject, "VOCAB", "LLAMA")
}


#endif //TFFINFER_LLMVOCABULARY_H
