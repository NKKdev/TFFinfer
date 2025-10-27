//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELCONFIGREADER_H
#define TFFINFER_MODELCONFIGREADER_H

// read_config.cpp
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>  // 第三方 JSON 库

using json = nlohmann::json;
namespace tff::core::model {
    class ModelConfigReader {
    public:
        struct Config {
            std::vector<std::string> architectures;
            bool valid = false;
        };

        static Config read(const std::string& filepath) {
            Config config;

            std::ifstream file(filepath);
            if (!file.is_open()) {
                std::cerr << "Error: Cannot open config file: " << filepath << std::endl;
                return config;
            }

            try {
                json j;
                file >> j;

                // 检查是否存在 "architectures" 字段
                if (!j.contains("architectures")) {
                    std::cerr << "Warning: 'architectures' field not found in " << filepath << std::endl;
                    return config;
                }

                // 检查是否为数组
                if (!j["architectures"].is_array() || j["architectures"].empty()) {
                    std::cerr << "Error: 'architectures' is not a non-empty array." << std::endl;
                    return config;
                }

                // 提取所有 architectures
                config.architectures = j["architectures"].get<std::vector<std::string>>();
                config.valid = true;

            } catch (const std::exception& e) {
                std::cerr << "Error parsing JSON: " << e.what() << std::endl;
            }

            return config;
        }
    };
}



#endif //TFFINFER_MODELCONFIGREADER_H