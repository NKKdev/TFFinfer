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
namespace tff::core::model {
    class ModelDetectorBase {
        public:
        ModelDetectorBase() = default;

        virtual ~ModelDetectorBase() = default;
    public:
        // 检查模型是否匹配此架构
        virtual bool matches(const std::vector<std::string> &architectures) const = 0;

        // 获取该模型的名称（用于日志）
        virtual const char* name() const = 0;

        // 创建该模型的加载器
        virtual std::shared_ptr<ModelLoaderBase> create_loader() = 0;

        // （可选）优先级，用于解决冲突
        virtual int priority() const { return 50; } // 默认优先级
    };
}

#endif //TFFINFER_MODELDETECTORBASE_H