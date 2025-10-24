//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_LLMVOCABULARY_H
#define TFFINFER_LLMVOCABULARY_H
#include "BaseDefine.h"
#include <unordered_map>

#include "mem/Tensor.h"

namespace tff::core::model {
    class LLMVocabulary {
    public:
        LLMVocabulary();

        ~LLMVocabulary();

    public:
        tff::core::model::TokenData _token_data;


    };
}


#endif //TFFINFER_LLMVOCABULARY_H
