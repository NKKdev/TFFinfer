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

    // 模型加载器接口
    class  ModelLoaderBase : public tff::module::ModuleObject{
    public:
        ModelLoaderBase() = default;
        ~ModelLoaderBase() override = default;

        // 【必须】从路径加载模型
        // 返回：是否成功
        virtual ModelLoadResult load_from_file(const std::vector<std::string> &model_files_name, bool use_mmap, bool check_tensors) = 0;

        // 【必须】获取模型上下文（包含权重、缓冲区等）
        virtual std::shared_ptr<tff::core::model::ModelContext> get_model_ctx() = 0;

        virtual ModelLoadResult convert_to_gguf(const std::string & output_path) {
            return ModelLoadResult::UNSUPPORTED_ARCH;
        }

        virtual const char * get_arch_name() const = 0;

        virtual const ModelConfig & get_model_config() const = 0;

        // 【可选】是否支持 mmap
        virtual bool supports_mmap() const { return true; }

        // 【可选】是否支持 mlock
        virtual bool supports_mlock() const { return true; }

        // 【可选】获取所有需要加载的张量名称列表（用于预分配）
        virtual std::vector<std::string> get_tensor_names() const = 0;
        //
        virtual const std::unordered_map<std::string, ModelWeight> &get_weight_map() const = 0;
        //
        virtual std::shared_ptr<FileMMap> get_file_map(const int &index) = 0;
    };
}




#endif //TFFINFER_MODELLOADERBASE_H