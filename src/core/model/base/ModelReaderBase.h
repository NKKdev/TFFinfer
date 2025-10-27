//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELREADERBASE_H
#define TFFINFER_MODELREADERBASE_H
#include "ModuleObject.h"

namespace tff::core::model {
    class ModelReaderBase : public tff::module::ModuleObject {
    public:
        ModelReaderBase() = default;

        ~ModelReaderBase() override = default;

    public:
        // 检查文件是否匹配此格式（通过路径或前几个字节）
        virtual bool matches(const std::string &path) const = 0;

        // 获取格式名称（用于日志）
        virtual const char *format_name() const = 0;

        // （可选）支持 mmap
        virtual bool supports_mmap() const { return false; }

        // （可选）支持流式加载
        virtual bool supports_streaming() const { return false; }
    };
}


#endif //TFFINFER_MODELREADERBASE_H
