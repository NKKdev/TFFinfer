//
// Created by nkk on 2025/11/10.
//

#include "LLAMACreator.h"

namespace tff::core::model {
    REGISTER_MODULE_OBJECT(LLAMACreator, ModelCreatorBase, MODEL_CREATOR_FLAG,
                           std::string(LLM_ARCH_NAMES.find(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA
                           )->second));
}
