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

        virtual ModelLoadResult load_from_file(const std::vector<std::string> &model_files_name,
            const tff::core::model::ModelConfig &params) = 0;

        virtual std::shared_ptr<tff::core::model::ModelContext> get_model_ctx() = 0;

        virtual ModelLoadResult convert_to_gguf(const std::string & output_path) {
            return ModelLoadResult::UNSUPPORTED_ARCH;
        }

        virtual const char * get_arch_name() const = 0;

        virtual const ModelConfig & get_model_config() const = 0;

        virtual bool supports_mmap() const { return true; }

        virtual std::vector<std::string> get_tensor_names() const = 0;
        //
        virtual const std::unordered_map<std::string, ModelWeight> &get_weight_map() const = 0;
        //
        virtual std::shared_ptr<FileMMap> get_file_map(const int &index) = 0;
    };
}




#endif //TFFINFER_MODELLOADERBASE_H