//
// Created by nkk on 2025/10/21.
//

#include "LLMModelLoader.h"
#include "FileLoader.h"
#include "Logger.h"
#include "mem/BaseDefine.h"

namespace tff::core::model {
    bool LLMModelLoader::load(const std::vector<std::string> &model_files_name, bool use_mmap, bool check_tensors) {
        bool bRet = true;
        for (size_t i = 0; i < model_files_name.size(); i++) {
            const std::string &model_file_name = model_files_name[i];
            this->_files_loader.insert(std::make_pair(i, std::make_unique<FileLoader>(model_file_name.c_str(), "rb")));
            const auto &it = this->_files_loader.find(i);
            _files_mmap.insert(
                std::make_pair(
                    i, std::make_unique<FileMMap>(
                        std::shared_ptr<FileLoader>(it->second.get()))));
            bRet &= this->check_file(it->second);
            bRet &= this->load_header(it->second, _gguf_ctx);
            bRet &= this->load_kv_meta(it->second, _gguf_ctx);
            bRet &= this->load_tensor_info(it->second, _gguf_ctx);
            if (!bRet) {
                return bRet;
            }
        }


        return bRet;
    }

    bool LLMModelLoader::check_file(const std::unique_ptr<FileLoader> &file_loader) const {
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

    bool LLMModelLoader::load_header(const std::unique_ptr<FileLoader> &file_loader,
                                     const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx) {
        bool bRet = true;
        bRet &= file_loader->read(gguf_ctx->_version);
        bRet &= gguf_ctx->check_version();
        bRet &= file_loader->read(this->_n_tensors);
        bRet &= file_loader->read(this->_n_kv);
        return bRet;
    }

    bool LLMModelLoader::load_kv_meta(const std::unique_ptr<FileLoader> &file_loader,
                                      const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx) {
        bool bRet = true;
        for (size_t i = 0; i < this->_n_kv; i++) {
            std::string key;
            tff::core::model::GGUFType meta_type = TFF_GGUF_TYPE_UNKNOWN;
            bool is_array = false;
            bRet &= file_loader->read(key);
            for (size_t j = 0; j < gguf_ctx->_kv.size(); j++) {
                if (gguf_ctx->_kv.contains(key)) {
                    tff::log::Logger::error("%s: key '%s' already exists\n", __func__, key.c_str());
                    bRet = false;
                }
            }
            size_t num_elements = 0;
            bRet &= file_loader->read(meta_type);
            if (meta_type == TFF_GGUF_TYPE_ARRAY) {
                file_loader->read(meta_type);
                file_loader->read(num_elements);
            }
#define GGUF_TYPE(t, cpp_type) case t: return this->handle_gguf_kv<t>(file_loader, gguf_ctx, key, num_elements, true);
            switch (meta_type) {
                TFF_GGUF_TYPES
                case TFF_GGUF_TYPE_ARRAY:
                default:
                    tff::log::Logger::error("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(),
                                            meta_type);
                    return false;
            }
#undef GGUF_TYPE
        }
        bRet &= gguf_ctx->_kv.size() == this->_n_kv;
        return bRet;
    }

    bool LLMModelLoader::load_tensor_info(const std::unique_ptr<FileLoader> &file_loader,
                                          const std::unique_ptr<tff::core::model::GGUFContext> &gguf_ctx) {
        bool bRet = true;
        for (size_t i = 0; i < this->_n_tensors; i++) {
            tff::core::model::GGUFTensorInfo tensor_info;
            file_loader->read(tensor_info._name);
            for (auto &j: gguf_ctx->_tensor_info) {
                if (tensor_info._name == j._name) {
                    bRet &= false;
                    return bRet;
                }
            }
            //
            uint32_t n_dims = 0;
            file_loader->read(n_dims);
            tensor_info._tensor_ptr->set_dims(n_dims);
            for (size_t j = 0; j < n_dims; j++) {
                int64_t shape_dim = 0;
                file_loader->read(shape_dim);
                tensor_info._tensor_ptr->set_shape(shape_dim, j);
            }
            //
            int32_t data_type = 0;
            file_loader->read(data_type);
            tensor_info._tensor_ptr->set_data_type(tff::core::memory::DataType(data_type));
            //
            file_loader->read(tensor_info._offset);
            gguf_ctx->_tensor_info.push_back(tensor_info);
        }
        bRet &= gguf_ctx->_tensor_info.size() == this->_n_tensors;
        //
        file_loader->file_aligned(gguf_ctx->_alignment);
        gguf_ctx->_offset = file_loader->tell();
        //
        for (size_t i = 0; i < gguf_ctx->_tensor_info.size(); ++i) {
            const GGUFTensorInfo &info = gguf_ctx->_tensor_info[i];
            if (info._offset != gguf_ctx->_size) {
                tff::log::Logger::error("%s: tensor '%s' has offset %d, expected %ld \n",
                                        __func__, info._name.c_str(), info._offset, gguf_ctx->_size);
                tff::log::Logger::error("%s: failed to read tensor data\n", __func__);
                bRet = false;
                return bRet;
            }
            size_t padded_size = TFF_PAD(info._tensor_ptr->get_bytes(), gguf_ctx->_alignment);
            if (SIZE_MAX - gguf_ctx->_size < padded_size) {
                tff::log::Logger::error("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                                        __func__, info._name.c_str(), gguf_ctx->_size, padded_size);
                bRet = false;
                return bRet;
            }
            gguf_ctx->_size += padded_size;
        }
        return bRet;
    }
}
