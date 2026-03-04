//
// Created by nkk on 2025/10/27.
//

#ifndef TFFINFER_LLAMALOADER_H
#define TFFINFER_LLAMALOADER_H
#include "global/ModelGlobalVar.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "model/BaseDefine.h"
#include "Logger.h"

namespace tff::core::model {
    /**
     * GGUF模型加载器
     */
    class GGUFLoader : public ModelLoaderBase {
    public:
        GGUFLoader() {
            _model_ctx = std::make_shared<tff::core::model::ModelContext>();
        };

        ~GGUFLoader() override = default;

    public:
        /**
         * 加载模型
         * @param model_files_name
         * @param params 模型配置参数 用户指定的参数
         * @return
         */
        ModelLoadResult load_from_file(const std::vector<std::string> &model_files_name,
                                       const tff::core::model::ModelConfig &params) override;

        /**
         * 转换模型为GGUF格式
         * @param output_path
         * @return
         */
        ModelLoadResult convert_to_gguf(const std::string &output_path) override {
            //todo
            return ModelLoadResult::UNSUPPORTED_ARCH;
        }

        /**
         * 获取模型架构名称
         * @return
         */
        const char *get_arch_name() const override;

        /**
         * 获取模型配置参数
         * @return
         */
        const ModelConfig &get_model_config() const override;

        /**
         * 是否支持mmap
         * @return
         */
        bool supports_mmap() const override;

        /**
         * 获取模型权重名称
         * @return
         */
        std::vector<std::string> get_tensor_names() const override;

        /**
         * 获取模型权重
         * @return
         */
        inline const std::unordered_map<std::string, ModelWeight> &get_weight_map() const {
            return this->_weight_map;
        };
        /**
         * 获取文件映射
         * @param index
         * @return
         */
        std::shared_ptr<FileMMap> get_file_map(const int &index) override {
            return this->_files_mmap[index];
        }

    protected:
        /**
         * 加载模型
         * @param model_files_name
         * @param params
         * @return
         */
        bool load(const std::vector<std::string> &model_files_name,
                  const tff::core::model::ModelConfig &params);

        /**
         * 检查文件是否符合当前格式
         * @param file_loader
         * @return
         */
        [[nodiscard]] bool check_file(const std::shared_ptr<FileLoader> &file_loader) const;

        /**
         * 加载模型权重头信息
         * @param file_loader 文件加载器
         * @param gguf_ctx 模型上下文
         * @return
         */
        bool load_header(const size_t &file_index,
                         const std::shared_ptr<FileLoader> &file_loader,
                         const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        /**
         * 加载模型权重元数据
         * @param file_loader 文件加载器
         * @param gguf_ctx 模型上下文
         * @return
         */
        bool load_kv_meta(const size_t &file_index,
                          const std::shared_ptr<FileLoader> &file_loader,
                          const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        /**
         *
         * @tparam T  数据类型
         * @param file_loader 文件加载器
         * @param gguf_ctx   模型上下文
         * @param key   键
         * @param n 数组长度
         * @param is_array 是否为数组
         * @return
         */
        template<typename T>
        bool load_array_meta(const std::shared_ptr<FileLoader> &file_loader,
                             const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx, const std::string &key,
                             const size_t &n,
                             const bool is_array = false) {
            if (is_array) {
                std::vector<T> value;
                try {
                    if (!file_loader->read(value, n)) {
                        return false;
                    }
                } catch (std::length_error &) {
                    tff::log::Logger::error("%s: encountered length_error while reading value for key '%s'\n", __func__,
                                            key.c_str());
                    return false;
                } catch (std::bad_alloc &) {
                    tff::log::Logger::error("%s: encountered bad_alloc error while reading value for key '%s'\n",
                                            __func__,
                                            key.c_str());
                    return false;
                }
                gguf_ctx->_kv.insert(std::make_pair(key, value));
            } else {
                T value;
                file_loader->read(value);
                gguf_ctx->_kv.insert(std::make_pair(key, value));
            }
            return true;
        };
        /**
         * 处理GGUF模型元信息键值对
         * @tparam T 数据类型
         * @param file_loader 文件加载器
         * @param gguf_ctx 模型上下文
         * @param key 键
         * @param n 数组长度
         * @param is_array 是否为数组
         * @return
         */
        template<GGUFType T>
        bool handle_gguf_kv(const std::shared_ptr<FileLoader> &file_loader,
                            const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx,
                            const std::string &key, const uint64_t &n, const bool is_array = false) {
            using cpp_type = typename gguf_type_to_cpp<T>::type;
            static_assert(!std::is_same_v<cpp_type, void>, "Unsupported GGUF type");
            return load_array_meta<cpp_type>(file_loader, gguf_ctx, key, n, is_array);
        }

        /**
         * 加载模型权重信息
         * @param file_index 文件索引 用于多文件的模型加载
         * @param file_loader 文件加载器
         * @param gguf_ctx 模型上下文
         * @return
         */
        bool load_tensor_info(const size_t &file_index,
                              const std::shared_ptr<FileLoader> &file_loader,
                              const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        /**
         * @brief 加载模型权重数据
         * @param file_loader  文件加载器
         * @param gguf_ctx 模型上下文
         * @return
         */
        bool load_tensor_data(const std::shared_ptr<FileLoader> &file_loader,
                              const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        /**
         * 加载模型配置信息
         * @param file_loader 文件加载器
         * @param ctx 模型上下文
         * @return
         */
        bool load_model_config(const std::shared_ptr<FileLoader> &file_loader,
                               const std::shared_ptr<tff::core::model::ModelContext> &ctx);

        /**
         * 获取模型上下文
         * @return
         */
        inline std::shared_ptr<tff::core::model::ModelContext> get_model_ctx() {
            return this->_model_ctx;
        };
        /**
         * 获取模型权重类型
         * @param tensor_name 模型权重名称
         * @return
         */
        tff::core::memory::ModelTensorType get_model_tensor_type(const std::string &tensor_name) const;

    private:
        int64_t _n_kv = 0;
        int64_t _n_tensors = 0;
        int _n_created = 0;

        uint64_t _n_elements = 0;
        size_t _n_bytes = 0;

        bool _use_mmap = false;
        bool _check_tensors{};
        bool _alloc = false;

        std::unordered_map<int, std::shared_ptr<FileLoader> > _files_loader;
        std::unordered_map<int, std::shared_ptr<FileMMap> > _files_mmap;
        std::unordered_map<int, std::shared_ptr<FileLock> > _files_lock;

        //
        tff::core::model::ModelConfig _model_config;

        //
        std::unordered_map<std::string, ModelWeight> _weight_map;
        //
        std::shared_ptr<tff::core::model::ModelContext> _model_ctx;
    };
}


#endif //TFFINFER_LLAMALOADER_H
