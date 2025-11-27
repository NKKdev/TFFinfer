//
// Created by nkk on 2025/11/18.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMWeightMemManager.h"

namespace tff::kernel {
    template<typename T>
    static void embedding_kernel_cpu(const int &model_file_index,
                                     const int &offset,
                                     const double &data_size,
                                     const std::shared_ptr<tff::core::model::ModelLoaderBase> &model_loader_ptr,
                                     std::vector<std::shared_ptr<tff::core::memory::Tensor>> &inputs,
                                     std::vector<std::shared_ptr<tff::core::memory::Tensor>> &outputs,
                                     std::shared_ptr<
                                         tff::core::runtime::LLMWeightMemManager> &mem_buffer_manager_ptr) {
        if (inputs.empty()) {
            tff::log::Logger::error("embedding kernel inputs is empty");
            return;
        }
        if (inputs.size() != 2) {
            tff::log::Logger::error("embedding kernel inputs size is invalid");
            return;
        }
        if (outputs.size() != 1) {
            tff::log::Logger::error("embedding kernel outputs size is invalid");
            return;
        }
        const auto& token_embed = *inputs.begin();
        auto token_embed_buffer = token_embed->get_buffer();
        const auto& batch_token = *inputs.rbegin();
        auto batch_token_buffer = batch_token->get_buffer();
        auto dequantize_callback = core::memory::type_traits_auto[token_embed->get_data_type()].dequantize_callback;
        //
        auto &output_tensor = *outputs.begin();
        auto mem_buffer = mem_buffer_manager_ptr->get_cpu_mapped_memory();
        output_tensor->set_buffer_data(mem_buffer.second, output_tensor->get_bytes(), mem_buffer.first);
        auto output_tensor_buffer = mem_buffer.second;

        for (size_t i = 0; i < batch_token->get_shape().size(); ++i) {
            auto &shape_dim = batch_token->get_shape()[i];
            for (size_t j = 0; j < shape_dim; ++j) {
                int32_t token_id = batch_token->at<int32_t>(j);
                tff::log::Logger::info("embedding token id=%d", token_id);
                auto quant_data_ptr = (const void *)((char *)token_embed_buffer->ptr() + token_id * token_embed->get_strides()[1]);
                auto float_data_ptr = (float *)((char *)output_tensor_buffer+ i * shape_dim + j);
                dequantize_callback(quant_data_ptr, float_data_ptr, token_embed->get_shape()[0]);
            }
        }
    }

    //
    template<typename T>
    void tff::kernel::Embedding<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        tff::log::Logger::info("layer node op:%s compute!", tff::kernel::Embedding<T>::get_op_name().c_str());
        const auto &name = get_param_value<std::string>(0, para_ptr);
        const auto model_file_index = get_param_value<uint16_t>(1, para_ptr);
        const auto offset = get_param_value<size_t>(2, para_ptr);
        const auto data_size = get_param_value<double>(3, para_ptr);
        const auto model_loader_ptr = get_param_value<std::shared_ptr<tff::core::model::ModelLoaderBase> >(4, para_ptr);
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor>> >(
            5, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor>> >(
            6, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(7, para_ptr);

        embedding_kernel_cpu<T>(model_file_index, offset, data_size, model_loader_ptr, input_tensors, output_tensors,
                                 mem_buffer_manager_ptr);
    }

    template class tff::kernel::Embedding<float>;
    template class tff::kernel::Embedding<double>;
    template class tff::kernel::Embedding<int32_t>;
    REGISTER_OP_OBJECT(Embedding, float);

    REGISTER_OP_OBJECT(Embedding, double);

    REGISTER_OP_OBJECT(Embedding, int32_t);
}
