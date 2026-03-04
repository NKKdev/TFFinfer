//
// Created by nkk on 2025/10/27.
//
#include <string>
#include "model/FileLoader.h"
#include "Logger.h"
#include "mem/BaseDefine.h"
#include "GGUFLoader.h"
#include "global/GlobalDefine.h"

namespace tff::core::model {
    REGISTER_MODULE_OBJECT(GGUFLoader, ModelLoaderBase, MODEL_LOADER_FLAG,
                           tff::core::model::to_string(tff::core::model::ModelFileFormat::TFF_MODEL_FORMAT_GGUF));

    tff::core::model::ModelLoadResult tff::core::model::GGUFLoader::load_from_file(
        const std::vector<std::string> &model_files_name, const tff::core::model::ModelConfig &params) {
        this->_model_config = params;
        if (this->load(model_files_name, params)) {
            return ModelLoadResult::SUCCESS;
        } else {
            return ModelLoadResult::FAILED;
        }
    }


    const char *tff::core::model::GGUFLoader::get_arch_name() const {
        return this->_model_config._arch_name.c_str();
    }

    const tff::core::model::ModelConfig &tff::core::model::GGUFLoader::get_model_config() const {
        return this->_model_config;
    }

    bool tff::core::model::GGUFLoader::supports_mmap() const {
        return ModelLoaderBase::supports_mmap();
    }

    std::vector<std::string> tff::core::model::GGUFLoader::get_tensor_names() const {
        std::vector<std::string> tensor_names;
        for (size_t i = 0; i < this->_n_tensors; i++) {
            auto &tensor_info = this->_model_ctx->_tensor_info[i];
            tensor_names.push_back(tensor_info._name);
        }
        return tensor_names;
    }


    bool GGUFLoader::load(const std::vector<std::string> &model_files_name,
                          const tff::core::model::ModelConfig &params) {
        bool bRet = true;
        for (size_t i = 0; i < model_files_name.size(); i++) {
            const std::string &model_file_name = model_files_name[i];
            this->_files_loader.insert(std::make_pair(i, std::make_shared<FileLoader>(model_file_name.c_str(), "rb")));
            const auto &it = this->_files_loader.find(i);
            if (params._use_mmap) {
                auto cpu_device_size = tff::core::device::get_device_size("CPU");
                bool is_numa = cpu_device_size > 1 ? true : false;
                _files_mmap.insert(
                    std::make_pair(
                        i, std::make_shared<FileMMap>(it->second, -1, is_numa)));
            }

            bRet &= this->check_file(it->second);
            bRet &= this->load_header(i, it->second, _model_ctx);
            bRet &= this->load_kv_meta(i, it->second, _model_ctx);
            bRet &= this->load_model_config(it->second, _model_ctx);
            bRet &= this->load_tensor_info(i, it->second, _model_ctx);


            if (!bRet) {
                return bRet;
            }
        }

        return bRet;
    }

    bool GGUFLoader::check_file(const std::shared_ptr<FileLoader> &file_loader) const {
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

    bool GGUFLoader::load_header(const size_t &file_index, const std::shared_ptr<FileLoader> &file_loader,
                                 const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx) {
        bool bRet = true;
        bRet &= file_loader->read(gguf_ctx->_version);
        bRet &= gguf_ctx->check_version();
        bRet &= file_loader->read(this->_n_tensors);
        bRet &= file_loader->read(this->_n_kv);
        return bRet;
    }

    bool GGUFLoader::load_kv_meta(const size_t &file_index, const std::shared_ptr<FileLoader> &file_loader,
                                  const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx) {
        bool bRet = true;
        for (size_t i = 0; i < this->_n_kv; i++) {
            std::string key;
            tff::core::model::GGUFType meta_type = TFF_GGUF_TYPE_UNKNOWN;
            bool is_array = false;
            bRet &= file_loader->read(key);
            //tff::log::Logger::info("kv key: %s \n", key.c_str());
            for (size_t j = 0; j < gguf_ctx->_kv.size(); j++) {
                if (gguf_ctx->_kv.contains(key)) {
                    tff::log::Logger::error("%s: key '%s' already exists\n", __func__, key.c_str());
                    bRet = false;
                }
            }
            size_t num_elements = 0;
            bRet &= file_loader->read(meta_type);
            if (meta_type == TFF_GGUF_TYPE_ARRAY) {
                is_array = true;
                file_loader->read(meta_type);
                file_loader->read(num_elements);
            }
#define GGUF_TYPE(t, cpp_type) case t: this->handle_gguf_kv<t>(file_loader, gguf_ctx, key, num_elements, is_array);break;
            switch (meta_type) {
                TFF_GGUF_TYPES
                case TFF_GGUF_TYPE_ARRAY:
                default:
                    tff::log::Logger::error("%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(),
                                            meta_type);
                    break;
            }
#undef GGUF_TYPE
        }
        bRet &= gguf_ctx->_kv.size() == this->_n_kv;
        return bRet;
    }

    bool GGUFLoader::load_tensor_info(const size_t &file_index, const std::shared_ptr<FileLoader> &file_loader,
                                      const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx) {
        bool bRet = true;
        for (size_t i = 0; i < this->_n_tensors; i++) {
            tff::core::model::GGUFTensorInfo tensor_info;
            file_loader->read(tensor_info._name);
            //
            //tff::log::Logger::info("tensor info name:%s", tensor_info._name.c_str());
            for (auto &j: gguf_ctx->_tensor_info) {
                if (tensor_info._name == j._name) {
                    bRet &= false;
                    return bRet;
                }
            }
            //
            uint32_t n_dims = 0;
            file_loader->read(n_dims);
            if (n_dims == 0 || n_dims > 4) {
                throw std::runtime_error("Invalid tensor shape dimension count");
                continue;
            }
            std::array<int64_t, MAX_TENSOR_DIM> shapes{};
            shapes.fill(1);
            for (size_t j = 0; j < n_dims; j++) {
                int64_t shape_dim = 0;
                file_loader->read(shape_dim);
                shapes[j] = shape_dim;
            }
            //
            int32_t data_type = 0;
            file_loader->read(data_type);
            auto type = static_cast<tff::core::memory::DataType>(data_type);
            tensor_info._tensor_ptr = std::make_shared<tff::core::memory::Tensor>( type,
                core::memory::MemoryType::TFF_MEM_TYPE_RESIDENT, shapes, true);
            tensor_info._tensor_ptr->set_tensor_type(this->get_model_tensor_type(tensor_info._name));

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
            auto &[_name, _offset, _byte_size, _tensor_ptr] = gguf_ctx->_tensor_info[i];
            if (_offset != gguf_ctx->_size) {
                tff::log::Logger::error("%s: tensor '%s' has offset %d, expected %ld \n",
                                        __func__, _name.c_str(), _offset, gguf_ctx->_size);
                tff::log::Logger::error("%s: failed to read tensor data\n", __func__);
                bRet = false;
                return bRet;
            }
            float padded_size = TFF_PAD(_tensor_ptr->get_bytes(), gguf_ctx->_alignment);
            _byte_size = padded_size;
            if (SIZE_MAX - gguf_ctx->_size < padded_size) {
                tff::log::Logger::error("%s: tensor '%s' size overflow, cannot accumulate size %zu + %zu\n",
                                        __func__, _name.c_str(), gguf_ctx->_size, padded_size);
                bRet = false;
                return bRet;
            }
            gguf_ctx->_size += padded_size;
            gguf_ctx->_max_tensor_byte_size = gguf_ctx->_max_tensor_byte_size < padded_size
                                                  ? padded_size
                                                  : gguf_ctx->_max_tensor_byte_size;

            //
            ModelWeight model_weight;
            model_weight._idx = file_index;
            model_weight._offs = gguf_ctx->_offset + _offset;
            model_weight._alignment_size = _byte_size;
            model_weight._byte_size = _tensor_ptr->get_bytes();
            model_weight._tensor_ptr = _tensor_ptr;
            this->_weight_map.emplace(_name, model_weight);
        }

        return bRet;
    }

    bool GGUFLoader::load_tensor_data(const std::shared_ptr<FileLoader> &file_loader,
                                      const std::shared_ptr<tff::core::model::ModelContext> &gguf_ctx) {
        if (this->_alloc) {
            std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> allocator =
                    tff::factory::ModuleFactory::instance()->create_shared<
                        tff::core::device::MemBufferAllocatorBaseObject>(
                        MEMORY_ALLOCATOR_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
            gguf_ctx->_data_memory_ptr = std::make_shared<tff::core::memory::Memory>(gguf_ctx->_size, nullptr, false,
                allocator);
            gguf_ctx->_data_memory_ptr->allocate();
            file_loader->read(gguf_ctx->_data_memory_ptr->ptr(), gguf_ctx->_size);
            bool bRet = true;
            for (size_t i = 0; i < this->_n_tensors; i++) {
                auto &tensor_info = gguf_ctx->_tensor_info[i];
                auto &tensor = tensor_info._tensor_ptr;
                auto data_ptr = (char *) (gguf_ctx->_data_memory_ptr->ptr()) + tensor_info._offset;
                tensor->set_buffer_data(data_ptr, tensor->get_bytes());
            }
        }
        return true;
    }

    bool GGUFLoader::load_model_config(const std::shared_ptr<FileLoader> &file_loader,
                                       const std::shared_ptr<tff::core::model::ModelContext> &ctx) {
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, std::string,
                       tff::core::model::ModelMetaKV::LLM_KV_GENERAL_ARCHITECTURE,
                       this->_model_config._arch_name);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_EMBEDDING_LENGTH,
                       this->_model_config._n_embd);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_BLOCK_COUNT,
                       this->_model_config._n_layer);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_EXPERT_COUNT,
                       this->_model_config._n_expert);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, uint32_t, tff::core::model::ModelMetaKV::LLM_KV_EXPERT_USED_COUNT,
                       this->_model_config._n_expert_used);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, bool, tff::core::model::ModelMetaKV::LLM_KV_ROPE_SCALING_FINETUNED,
                       this->_model_config._rope_fine_tuned);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, uint32_t,
                       tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_KEY_LENGTH,
                       this->_model_config._n_embd_head_k);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, uint32_t,
                       tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_VALUE_LENGTH,
                       this->_model_config._n_embd_head_v);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, uint32_t,
                       tff::core::model::ModelMetaKV::LLM_KV_ROPE_DIMENSION_COUNT,
                       this->_model_config._n_rot);
        //
        LOAD_KEY_VALUES(ctx, ModelContext::BasicType, uint32_t,
                        tff::core::model::ModelMetaKV::LLM_KV_FEED_FORWARD_LENGTH,
                        this->_model_config._n_ff_arr);
        LOAD_KEY_VALUES(ctx, ModelContext::BasicType, uint32_t,
                        tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT,
                        this->_model_config._n_head_arr);
        LOAD_KEY_VALUES(ctx, ModelContext::BasicType, uint32_t,
                        tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_HEAD_COUNT_KV,
                        this->_model_config._n_head_kv_arr);
        LOAD_KEY_VALUE(ctx, ModelContext::BasicType, float,
                       tff::core::model::ModelMetaKV::LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,
                       this->_model_config._f_norm_rms_eps);

        return true;
    }

    std::vector<std::string> split(const std::string &s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);
        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    bool is_digits(const std::string &str) {
        return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
    }

    bool matches_pattern(const std::string &raw_name, const std::string &pattern) {
        auto name_tokens = split(raw_name, '.');
        auto pat_tokens = split(pattern, '.');
        if (name_tokens.size() != pat_tokens.size()) {
            return false;
        }
        for (size_t i = 0; i < name_tokens.size(); ++i) {
            if (pat_tokens[i] == "%d") {
                if (!is_digits(name_tokens[i])) {
                    return false;
                }
            } else {
                if (name_tokens[i] != pat_tokens[i]) {
                    return false;
                }
            }
        }
        return true;
    }

    tff::core::memory::ModelTensorType GGUFLoader::get_model_tensor_type(const std::string &tensor_name) const {
        std::string raw_name = tensor_name;
        const std::string weight_suffix = ".weight";
        if (raw_name.size() >= weight_suffix.size() &&
            raw_name.substr(raw_name.size() - weight_suffix.size()) == weight_suffix) {
            raw_name.resize(raw_name.size() - weight_suffix.size());
        }

        auto arch = tff::core::model::from_string(this->_model_config._arch_name);
        auto it = LLM_TENSOR_NAMES.find(arch);
        if (it == LLM_TENSOR_NAMES.end()) {
            return tff::core::memory::ModelTensorType::LLM_TENSOR_TYPE_UNKNOWN;
        }
        const auto &tensor_map = it->second;
        for (const auto &[tensor_type, pattern_str]: tensor_map) {
            std::string pattern(pattern_str);

            if (matches_pattern(raw_name, pattern)) {
                return tensor_type;
            }
        }
        return tff::core::memory::ModelTensorType::LLM_TENSOR_TYPE_UNKNOWN;
    }

    // tff::core::memory::ModelTensorType GGUFLoader::get_model_tensor_type(const std::string &tensor_name) const {
    //     std::string tmp = tensor_name.substr(0, tensor_name.find(".weight"));
    //     auto tensor_names_map = LLM_TENSOR_NAMES.find(tff::core::model::from_string(this->_model_config._arch_name));
    //     if (tensor_names_map == LLM_TENSOR_NAMES.end()) {
    //         return memory::ModelTensorType::LLM_TENSOR_TYPE_UNKNOWN;
    //     }
    //     for (auto &tensor_type: tensor_names_map->second) {
    //         auto str = std::string(tensor_type.second);
    //         auto pos = str.find_last_of(".");
    //         if (pos == std::string::npos) {
    //             if (str.find(tmp) != std::string::npos || tmp.find(str) != std::string::npos) {
    //                 return tensor_type.first;
    //             }
    //         } else {
    //             std::string substr = str.substr(str.find_last_of("."), str.size() - pos);
    //             if (substr.find(tmp) != std::string::npos || tmp.find(substr) != std::string::npos) {
    //                 return tensor_type.first;
    //             }
    //         }
    //     }
    //     return tff::core::memory::ModelTensorType::LLM_TENSOR_TYPE_UNKNOWN;
    // }
}
