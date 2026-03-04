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
    /**
     * 模型检测器基类
     */
    class ModelDetectorBase {
        public:
        ModelDetectorBase() = default;

        virtual ~ModelDetectorBase() = default;
    public:
        /**
         * 模型格式是否匹配
         * @param file_format 文件格式
         * @return
         */
        [[nodiscard]] virtual bool matches(const std::string &file_format) const = 0;
        /**
         * 模型名称
         * @return
         */
        [[nodiscard]] virtual const char* name() const = 0;
        /**
         * 创建模型加载器
         * @return
         */
        virtual std::shared_ptr<ModelLoaderBase> create_loader() = 0;

    };
}

#endif //TFFINFER_MODELDETECTORBASE_H