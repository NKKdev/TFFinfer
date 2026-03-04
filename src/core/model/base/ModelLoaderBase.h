//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_MODELLOADERBASE_H
#define TFFINFER_MODELLOADERBASE_H
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "ModuleObject.h"
#include "graph/Graph.h"
#include "graph/GraphNode.h"
#include "model/BaseDefine.h"
#include "model/FileLoader.h"

namespace tff::core::model {
    enum class ModelLoadResult {
        SUCCESS,
        FILE_NOT_FOUND,
        INVALID_FORMAT,
        UNSUPPORTED_ARCH,
        OUT_OF_MEMORY,
        FAILED
    };

    /**
     * 模型加载器基类
     */
    class ModelLoaderBase : public tff::module::ModuleObject {
    public:
        ModelLoaderBase() = default;

        ~ModelLoaderBase() override = default;

    public:
        /**
         * 加载模型
         * @param model_files_name 模型文件名
         * @param params 模型参数
         * @return 加载结果
         */
        virtual ModelLoadResult load_from_file(const std::vector<std::string> &model_files_name,
                                               const tff::core::model::ModelConfig &params) = 0;

        /**
         * 获取模型上下文
         * @return 模型上下文
         */
        virtual std::shared_ptr<tff::core::model::ModelContext> get_model_ctx() = 0;

        /**
         * 转换模型为gguf格式
         * @param output_path 输出路径
         * @return 转换结果
         */
        virtual ModelLoadResult convert_to_gguf(const std::string &output_path) {
            //todo
            return ModelLoadResult::UNSUPPORTED_ARCH;
        }

        /**
         * @brief 获取模型架构名称
         * @return
         */
        virtual const char *get_arch_name() const = 0;

        /**
         * @brief 获取模型配置参数
         * @return
         */
        virtual const ModelConfig &get_model_config() const = 0;

        /**
         * @brief 获取模型是否支持mmap
         * @return
         */
        virtual bool supports_mmap() const { return true; }
        /**
         * @brief 获取模型权重名称
         * @return
         */
        virtual std::vector<std::string> get_tensor_names() const = 0;

        /**
         * @brief 获取模型权重映射
         * @return
         */
        virtual const std::unordered_map<std::string, ModelWeight> &get_weight_map() const = 0;

        /**
         * @brief 获取文件映射
         * @param index 文件索引
         * @return
         */
        virtual std::shared_ptr<FileMMap> get_file_map(const int &index) = 0;
    };
}


#endif //TFFINFER_MODELLOADERBASE_H
