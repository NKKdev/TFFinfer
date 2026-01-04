//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELDETECTORBASE_H
#define TFFINFER_MODELDETECTORBASE_H

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "ModuleObject.h"
#include "model/BaseDefine.h"
#include "ModelLoaderBase.h"
#include "ModelCreatorBase.h"
namespace tff::core::model {
    class ModelDetectorBase {
        public:
        ModelDetectorBase() = default;

        virtual ~ModelDetectorBase() = default;
    public:
        // 检查模型是否匹配此格式
        virtual bool matches(const std::string &file_format) const = 0;
        // 获取该模型格式的名称
        virtual const char* name() const = 0;
        // 创建该模型的加载器
        virtual std::shared_ptr<ModelLoaderBase> create_loader() = 0;

    };
}

#endif //TFFINFER_MODELDETECTORBASE_H