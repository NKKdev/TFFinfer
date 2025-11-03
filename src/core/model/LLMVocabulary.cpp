//
// Created by nkk on 2025/10/21.
//

#include "LLMVocabulary.h"
#include "BaseDefine.h"
#include "../global/ModelGlobalVar.h"
#include "Logger.h"
#include "util.h"
#include "unicode.h"

namespace tff::core::model {
    bool tff::core::model::LLMLLaMaVocabulary::load_vocabulary(
        const std::shared_ptr<tff::core::model::ModelLoaderBase> &model_loader) {
        bool bRet = true;
        this->_model_loader = model_loader;
        const auto &ctx = this->_model_loader->get_model_context();
        LOAD_KEY_VALUE(ModelContext::BasicType, std::string, tff::core::model::ModelMetaKV::LLM_KV_GENERAL_ARCHITECTURE,
                               this->_arch_name);
        LOAD_KEY_VALUE(ModelContext::BasicType, std::string, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_MODEL,
                       this->_tokenizer_model);
        LOAD_KEY_VALUE(ModelContext::BasicType, std::string, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_PRE,
                       this->_tokenizer_pre);
        LOAD_KEY_VALUE(ModelContext::BasicType, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT,
                       this->_n_token_types);
        LOAD_KEY_VALUE(ModelContext::BasicType, bool, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_ADD_PREFIX,
                       this->_add_space_prefix);
        LOAD_KEY_VALUE(ModelContext::BasicType, bool, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_REMOVE_EXTRA_WS,
                       this->_remove_extra_whitespaces);

        if (this->_tokenizer_pre == "smollm") {
            this->_vacabulary_pre_type = TFF_VOCAB_PRE_TYPE_SMOLLM;
        }
        const auto &type = LLM_TOKENIZER_MODEL_VOCAB_TYPE.find(this->_tokenizer_model)->second;
        this->_tokenizer =
                tff::factory::ModuleFactory::instance()->create_shared<tff::core::model::LLMTokenizerBase>(
                    "TOKENIZER", get_tokenizer_name(type).data());


        bRet &= this->load_bpe();
        //
        bRet &= this->load_token_data();

        return bRet;
    }

    int32_t LLMLLaMaVocabulary::get_rank(const std::string &token_left, const std::string &token_right) const {
        auto it = this->_bpe_ranks.find(std::make_pair(token_left, token_right));
        if (it == this->_bpe_ranks.end()) {
            return -1;
        }

        return it->second;
    }

    bool LLMLLaMaVocabulary::load_bpe() {
        const auto &ctx = _model_loader->get_model_context();
        //
        std::vector<std::string> merge_vector;
        LOAD_KEY_VALUES(std::vector<std::string>,std::string, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_MERGES,
                        merge_vector);

        for (int i = 0; i < merge_vector.size(); i++) {
            const std::string &word = merge_vector[i];

            std::string first;
            std::string second;

            if (const size_t pos = word.find(' ', 1); pos != std::string::npos) {
                first = word.substr(0, pos);
                second = word.substr(pos + 1);
            }

            _bpe_ranks.emplace(std::make_pair(first, second), i);
        }
        return true;
    }

    bool LLMLLaMaVocabulary::load_token_data() {
        const auto &ctx = _model_loader->get_model_context();
        //
        std::vector<std::string> token_data;
        LOAD_KEY_VALUES(std::vector<std::string>, std::string, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_LIST,
                        token_data);
        //
        std::vector<float> token_scores;
        LOAD_KEY_VALUES(ModelContext::BasicType,float, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_SCORES, token_scores);
        //
        std::vector<uint32_t> token_types;
        LOAD_KEY_VALUES(ModelContext::BasicType,uint32_t, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_TOKEN_TYPE,
                        token_types);
        if (token_data.empty()) {
            return false;
        }
        this->_id_to_token.resize(token_data.size());
        for (int i = 0; i < token_data.size(); i++) {
            const std::string &word = token_data[i];
            this->_token_to_id[word] = i;
            auto &[_text, _attribute, _score] = this->_id_to_token[i];
            _text = word;
            _score = token_scores.size() != token_data.size() ? 0 : token_scores[i];
            _attribute = token_types.empty()
                             ? TokenAttribute::TFF_TOKEN_ATTR_NORMAL
                             : static_cast<TokenAttribute>(1 << static_cast<int>(token_types[i]));
        }
        //special_token
        this->process_special_tokens();
        //token_cache
        size_t size_cache = 0;
        std::vector<std::string> cache(this->_id_to_token.size());

        for (uint32_t id = 0; id < cache.size(); ++id) {
            cache[id] = this->token_to_string(id, true);
            size_cache += cache[id].size();
        }
        std::swap(this->_cache_token_to_piece, cache);

        return true;
    }

    void LLMLLaMaVocabulary::tokenize(const std::string &raw_text, std::vector<int32_t> &token_vec,
                                      bool add_special, bool parse_special) const {
        if (this->_tokenizer == nullptr) {
            tff::log::Logger::error("tokenizer not init! raw_text: %s tokenize failed!!\n", raw_text.c_str());
            return;
        }
        this->_tokenizer->tokenize(raw_text, token_vec, this);
    }

    void LLMLLaMaVocabulary::process_special_tokens() {
        const auto &ctx = _model_loader->get_model_context();
        for (auto &pair: LLM_SPECIAL_TOKENS) {
            //
            LOAD_KEY_VALUE(ModelContext::BasicType,uint32_t, pair.first, pair.second);
        }

        LOAD_KEY_VALUE(ModelContext::BasicType,bool, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_ADD_BOS, this->_add_bos);
        LOAD_KEY_VALUE(ModelContext::BasicType,bool, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_ADD_EOS, this->_add_eos);
        LOAD_KEY_VALUE(ModelContext::BasicType,bool, tff::core::model::ModelMetaKV::LLM_KV_TOKENIZER_ADD_SEP, this->_add_sep);

#define UPDATE_SPECIAL_TOKEN(special_token_type, key, value) \
    if (LLM_SPECIAL_TOKENS[special_token_type] == LLAMA_TOKEN_NULL) { \
        const auto &special_token_vec = LLM_SPECIAL_TOKEN_STRING.find(special_token_type)->second;\
        auto iter = std::find(special_token_vec.begin(), special_token_vec.end(),key);\
        if (iter != special_token_vec.end()) { \
            LLM_SPECIAL_TOKENS[special_token_type] = value; \
            auto &token_data = this->_id_to_token[value]; \
            if ((token_data._attribute & TokenAttribute::TFF_TOKEN_ATTR_CONTROL) == 0) { \
                token_data._attribute = TokenAttribute::TFF_TOKEN_ATTR_CONTROL; \
            } \
        } \
    }
        for (auto &pair: this->_token_to_id) {
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_EOT_ID, pair.first, pair.second);
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_EOM_ID, pair.first, pair.second);
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_FIM_PRE_ID, pair.first, pair.second);
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_FIM_SUF_ID, pair.first, pair.second);
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_FIM_MID_ID, pair.first, pair.second);
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_FIM_PAD_ID, pair.first, pair.second);
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_FIM_REP_ID, pair.first, pair.second);
            UPDATE_SPECIAL_TOKEN(LLM_KV_TOKENIZER_FIM_SEP_ID, pair.first, pair.second);
        }
#undef UPDATE_SPECIAL_TOKEN

        //user_defined_tokens
        this->process_user_defined_tokens();
        //eog_token;
        this->process_eog_tokens();
        //
        for (int32_t id = 0; id < this->_id_to_token.size(); ++id) {
            auto &token_data = this->_id_to_token[id];
            if (token_data._attribute & (TFF_TOKEN_ATTR_CONTROL | TFF_TOKEN_ATTR_USER_DEFINED |
                                         TFF_TOKEN_ATTR_UNKNOWN)) {
                this->_cache_special_tokens.push_back(id);
            }
        }

        std::sort(this->_cache_special_tokens.begin(), this->_cache_special_tokens.end(),
                  [&](const int32_t a, const int32_t b) {
                      return this->_id_to_token[a]._text.size() > this->_id_to_token[b]._text.size();
                  }
        );
        //linefeed_token;
        std::vector<int32_t> token_vec;
        this->tokenize(std::string("\n"), token_vec, false, false);

        if (token_vec.empty()) {
            tff::log::Logger::info("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            this->_linefeed_id = LLM_SPECIAL_TOKENS[LLM_KV_TOKENIZER_PAD_ID];
        } else {
            this->_linefeed_id = token_vec[0];
        }
    }

    void LLMLLaMaVocabulary::process_eog_tokens() {
        this->_special_eog_ids.clear();
#define UPDATE_EOG_TOKEN(special_token_type) \
        if (LLM_SPECIAL_TOKENS[special_token_type] != LLAMA_TOKEN_NULL && \
        this->_special_eog_ids.count(LLM_SPECIAL_TOKENS[special_token_type]) == 0) {\
            this->_special_eog_ids.insert(LLM_SPECIAL_TOKENS[special_token_type]);\
        }

        UPDATE_EOG_TOKEN(LLM_KV_TOKENIZER_FIM_PAD_ID);
        UPDATE_EOG_TOKEN(LLM_KV_TOKENIZER_FIM_REP_ID);
        UPDATE_EOG_TOKEN(LLM_KV_TOKENIZER_FIM_SEP_ID);

        UPDATE_EOG_TOKEN(LLM_KV_TOKENIZER_EOS_ID);
        UPDATE_EOG_TOKEN(LLM_KV_TOKENIZER_EOT_ID);
        UPDATE_EOG_TOKEN(LLM_KV_TOKENIZER_EOM_ID);

#undef UPDATE_EOG_TOKEN
        for (const auto &t: this->_token_to_id) {
            if (false
                || t.first == "<|eot_id|>"
                || t.first == "<|im_end|>"
                || t.first == "<|end|>"
                || t.first == "<|return|>" // o200k_harmony
                || t.first == "<|call|>" // o200k_harmony
                || t.first == "<end_of_turn>"
                || t.first == "<|endoftext|>"
                || t.first == "<|eom_id|>"
                || t.first == "<EOT>"
                || t.first == "_<EOT>"
                || t.first == "<|end_of_text|>"
                || t.first == "<end_of_utterance>" // smoldocling
            ) {
                this->_special_eog_ids.insert(t.second);
                if ((this->_id_to_token[t.second]._attribute & TFF_TOKEN_ATTR_CONTROL) == 0) {
                    tff::log::Logger::warning(
                        "%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                        __func__, t.second, t.first.c_str());
                    this->_id_to_token[t.second]._attribute = TFF_TOKEN_ATTR_CONTROL;
                }
            } else {
                // token is control, but not marked as EOG -> print a debug log
                if (this->_id_to_token[t.second]._attribute & TFF_TOKEN_ATTR_CONTROL && this->_special_eog_ids.
                    count(t.second) == 0) {
                    tff::log::Logger::warning("%s: control token: %6d '%s' is not marked as EOG\n",
                                              __func__, t.second, t.first.c_str());
                }
            }
        }

        //
        bool has_return = false;
        bool has_call = false;
        bool has_end = false;

        int32_t end_id = LLAMA_TOKEN_NULL;

        for (auto tid: this->_special_eog_ids) {
            tff::log::Logger::info("%s:   - %d ('%s')\n", __func__, tid, this->_id_to_token[tid]._text.c_str());

            if (this->_id_to_token[tid]._text == "<|return|>") {
                has_return = true;
            } else if (this->_id_to_token[tid]._text == "<|call|>") {
                has_call = true;
            } else if (this->_id_to_token[tid]._text == "<|end|>") {
                has_end = true;
                end_id = tid;
            }
        }

        if (has_return && has_call && has_end) {
            this->_special_eog_ids.erase(end_id);
            this->_id_to_token[end_id]._attribute = TFF_TOKEN_ATTR_USER_DEFINED;
            tff::log::Logger::info(
                "%s: special_eog_ids contains both '<|return|>' and '<|call|>' tokens, removing '<|end|>' token from EOG list\n",
                __func__);
        }
    }

    //
    void LLMLLaMaVocabulary::process_user_defined_tokens() {
        for (const auto &t: this->_token_to_id) {
            if (t.first == "<|channel|>" || t.first == "<|message|>" || t.first == "<|start|>" || t.first ==
                "<|constrain|>") {
                this->_id_to_token[t.second]._attribute = TFF_TOKEN_ATTR_USER_DEFINED;
            }
        }
    }

    static int32_t check_space(const char *token, size_t size, int32_t length, int32_t lstrip) {
        for (size_t i = 0; i < lstrip && size && *token == ' '; i++) {
            token++;
            size--;
            if (length < static_cast<int32_t>(size)) {
                return -static_cast<int32_t>(size);
            }
        }
        return length;
    }

    //
    std::string LLMLLaMaVocabulary::token_to_string(const int32_t &token, bool special) {
        const auto &text = this->_id_to_token[token]._text;
        std::string piece;
        piece.resize(text.size()); // using string internal cache
        const int n_chars = this->token_to_string(token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = this->token_to_string(token, &piece[0], piece.size(), 0, special);
            if (check < 0) {
                return "";
            }
        } else {
            piece.resize(n_chars);
        }

        return piece;
    }

    //
    int32_t LLMLLaMaVocabulary::token_to_string(int32_t token, char *buf, int32_t length, int32_t lstrip,
                                                bool special) {
        int32_t nCheckRet = 0;
        const auto &attr = this->_id_to_token[token]._attribute;
        const auto &text = this->_id_to_token[token]._text;
        if (!special && (attr & (TFF_TOKEN_ATTR_UNKNOWN | TFF_TOKEN_ATTR_CONTROL))) {
            return nCheckRet;
        }
#define CHECK_SPACE(token_data, size) \
        nCheckRet = check_space(token_data, size, length, lstrip);\
        if (nCheckRet < 0) {\
            return nCheckRet;\
        }\
        memcpy(buf, token_data, size);

        auto *text_data = text.data();
        auto size = text.size();
        //
        if (!this->_cache_token_to_piece.empty()) {
            const auto &cache_token = this->_cache_token_to_piece.at(token);
            CHECK_SPACE(text_data, size);
            return size;
        }
        //
        if (attr & (TFF_TOKEN_ATTR_UNKNOWN | TFF_TOKEN_ATTR_CONTROL | TFF_TOKEN_ATTR_USER_DEFINED)) {
            CHECK_SPACE(text_data, size);
        }
        if (attr & TFF_TOKEN_ATTR_NORMAL) {
            std::string result = decode_text(text);
            CHECK_SPACE(result.data(), result.size());
            return result.size();
        }
        return nCheckRet;
    }
};
