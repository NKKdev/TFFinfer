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
    class LLAMALoader : public ModelLoaderBase {
    public:
        LLAMALoader() {
            _model_ctx = std::make_shared<tff::core::model::ModelContext>();
        };

        ~LLAMALoader() override = default;

    public:
        //
        ModelLoadResult load_from_file(const std::vector<std::string> &model_files_name, bool use_mmap,
                                       bool check_tensors) override;

        ModelLoadResult convert_to_gguf(const std::string &output_path) override {
            return ModelLoadResult::UNSUPPORTED_ARCH;
        }

        const char *get_arch_name() const override;

        const ModelConfig &get_model_config() const override;

        bool supports_mmap() const override;

        bool supports_mlock() const override;

        std::vector<std::string> get_tensor_names() const override;

        //
        inline const std::unordered_map<std::string, ModelWeight> &get_weight_map() const {
            return this->_weight_map;
        };
        //
        std::shared_ptr<FileMMap> get_file_map(const int &index) override {
            return this->_files_mmap[index];
        }

    protected:
        bool load(const std::vector<std::string> &model_files_name,
                  bool use_mmap,
                  bool check_tensors);

        //
        [[nodiscard]] bool check_file(const std::shared_ptr<FileLoader> &file_loader) const;

        //
        bool load_header(const size_t &file_index,
                         const std::shared_ptr<FileLoader> &file_loader,
                         const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        //
        bool load_kv_meta(const size_t &file_index,
                          const std::shared_ptr<FileLoader> &file_loader,
                          const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        //
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
        //
        template<GGUFType T>
        bool handle_gguf_kv(const std::shared_ptr<FileLoader> &file_loader,
                            const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx,
                            const std::string &key, const uint64_t &n, const bool is_array = false) {
            using cpp_type = typename gguf_type_to_cpp<T>::type;
            static_assert(!std::is_same_v<cpp_type, void>, "Unsupported GGUF type");
            return load_array_meta<cpp_type>(file_loader, gguf_ctx, key, n, is_array);
        }

        //
        bool load_tensor_info(const size_t &file_index,
                              const std::shared_ptr<FileLoader> &file_loader,
                              const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        //
        bool load_tensor_data(const std::shared_ptr<FileLoader> &file_loader,
                              const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx);

        //
        bool load_model_config(const std::shared_ptr<FileLoader> &file_loader,
                               const std::shared_ptr<tff::core::model::ModelContext> &ctx);

        //
        inline std::shared_ptr<tff::core::model::ModelContext> get_model_ctx() {
            return this->_model_ctx;
        };
        //
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
