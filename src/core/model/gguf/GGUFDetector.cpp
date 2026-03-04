//
// Created by nkk on 2025/10/27.
//

#include "GGUFDetector.h"
#include "global/GlobalDefine.h"
namespace tff::core::model {
    REGISTER_MODULE_OBJECT(GGUFDetector, ModelDetectorBase,MODEL_DETECTOR_FLAG,
                           tff::core::model::to_string(tff::core::model::ModelFileFormat::TFF_MODEL_FORMAT_GGUF));
}