//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_LLAMAREADER_H
#define TFFINFER_LLAMAREADER_H
#include "model/base/ModelReaderBase.h"

namespace tff::core::model {
    class LLAMAReader : public ModelReaderBase {
    public:
        LLAMAReader() = default;

        ~LLAMAReader() override = default;

    public:
        // 检查文件是否匹配此格式（通过路径或前几个字节）
        bool matches(const std::string &path) const override;

        // 获取格式名称（用于日志）
        const char *format_name() const override;

        // （可选）支持 mmap
        bool supports_mmap() const override { return false; }

        // （可选）支持流式加载
        bool supports_streaming() const override { return false; }
    };
}

#endif //TFFINFER_LLAMAREADER_H
