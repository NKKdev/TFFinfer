//
// Created by nkk on 2025/10/21.
//

#ifndef TFFINFER_LLMMODELLOADER_H
#define TFFINFER_LLMMODELLOADER_H
#include "ModuleFactory.h"
#include "ModuleObject.h"
#include "FileLoader.h"
#include "BaseDefine.h"
#include "Logger.h"

namespace tff::core::model {
    class LLMModelLoader : public tff::module::ModuleObject {
    public:
        LLMModelLoader() : _check_tensors(false) {
            _no_alloc = true;
        } ;

        ~LLMModelLoader() override = default;

    public:
        bool load(const std::vector<std::string> &model_files_name,
                  bool use_mmap,
                  bool check_tensors);

        //
        [[nodiscard]] bool check_file(const std::unique_ptr<FileLoader> &file_loader) const;

        //
        bool load_header(const std::unique_ptr<FileLoader> &file_loader,
        const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx);

        //
        bool load_kv_meta(const std::unique_ptr<FileLoader> &file_loader,
        const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx);

        //
        template<typename T>
        bool load_array_meta(const std::unique_ptr<FileLoader> &file_loader,
        const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx, const std::string &key, const size_t &n,
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
        bool handle_gguf_kv(const std::unique_ptr<FileLoader> &file_loader,
                            const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx,
                            const std::string &key, const uint64_t &n, const bool is_array = false) {
            using cpp_type = typename gguf_type_to_cpp<T>::type;
            static_assert(!std::is_same_v<cpp_type, void>, "Unsupported GGUF type");
            return load_array_meta<cpp_type>(file_loader, gguf_ctx, key, n, is_array);
        }

        //
        bool load_tensor_info(const std::unique_ptr<FileLoader> &file_loader,
        const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx);

        //
        inline std::unique_ptr<tff::core::model::GGUFContext> &get_gguf_ctx() {
            return this->_gguf_ctx;
        };


    public:
        int _n_kv = 0;
        int _n_tensors = 0;
        int _n_created = 0;

        uint64_t _n_elements = 0;
        size_t _n_bytes = 0;

        bool _use_mmap = false;
        bool _check_tensors;
        bool _no_alloc = true;

        std::unordered_map<int, std::unique_ptr<FileLoader> > _files_loader;
        std::unordered_map<int, std::unique_ptr<FileMMap> > _files_mmap;
        std::unordered_map<int, std::unique_ptr<FileLock> > _files_lock;

    protected:
        std::unique_ptr<tff::core::model::GGUFContext> _gguf_ctx;
    };

    REGISTER_MODULE_OBJECT(LLMModelLoader, "MODEL", "LLAMA")
}


#endif //TFFINFER_LLMMODELLOADER_H
