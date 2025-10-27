//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELREADERREGISTRY_H
#define TFFINFER_MODELREADERREGISTRY_H
#include <vector>
#include <memory>
#include "ModelReaderBase.h"
#include "ModuleFactory.h"

namespace tff::core::model {
    class ModelDetectyorRegistry {
    public:
        static ModelDetectyorRegistry &get() {
            static ModelDetectyorRegistry instance;
            return instance;
        }


        // 自动探测并返回合适的读取器
        std::shared_ptr<ModelReaderBase> &detect_format(const std::string &path) const {
            auto readers = tff::factory::ModuleFactory::instance()->create_shared_list(MODEL_READER_TYPE);
            for (auto &reader: readers) {
                auto reader_ptr = std::dynamic_pointer_cast<ModelReaderBase>(reader.second.front()());
                if (reader_ptr) {
                    if (reader_ptr->matches(path)) {
                        return reader_ptr;
                    }
                }
            }
            return nullptr;
        }

        // 获取所有支持的格式（用于日志）
        std::vector<const char *> get_supported_formats() const {
            std::vector<const char *> names;
            auto readers = tff::factory::ModuleFactory::instance()->create_shared_list(MODEL_READER_TYPE);
            for (auto &reader: readers) {
                names.push_back(reader.first.c_str());
            }
            return names;
        }
    };
}

#endif //TFFINFER_MODELREADERREGISTRY_H
