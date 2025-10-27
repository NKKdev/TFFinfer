//
// Created by nkk on 2025/10/27.
//

#include "LLAMAReader.h"
#include "model/FileLoader.h"
#include "model/GGUFDef.h"
bool tff::core::model::LLAMAReader::matches(const std::string &model_file_name) const {
    auto file_loader = std::make_unique<FileLoader>(model_file_name.c_str(), "rb");
    std::vector<char> buffer_format;
    file_loader->read<char>(buffer_format, 4);
    for (uint32_t i = 0; i < buffer_format.size(); i++) {
        if (buffer_format[i] != GGUF_MAGIC[i]) {
            char c0 = isprint(buffer_format[0]) ? buffer_format[0] : '?';
            char c1 = isprint(buffer_format[1]) ? buffer_format[1] : '?';
            char c2 = isprint(buffer_format[2]) ? buffer_format[2] : '?';
            char c3 = isprint(buffer_format[3]) ? buffer_format[3] : '?';
            tff::log::Logger::error("%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, c0, c1,
                                    c2, c3);
            return false;
        }
    }
    return true;
}

const char * tff::core::model::LLAMAReader::format_name() const {
    return GGUF_MAGIC;
}
