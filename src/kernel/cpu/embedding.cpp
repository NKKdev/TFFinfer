//
// Created by nkk on 2025/11/18.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/MemManager.h"
#include "kernel/include/BaseDefine.h"

namespace tff::kernel {
    template<typename T>
    static void embedding_kernel_cpu(
        std::shared_ptr<tff::core::memory::Tensor> &token_embed,
        std::shared_ptr<tff::core::memory::Tensor> &batch_token,
        std::shared_ptr<tff::core::memory::Tensor> &outputs) {
        if (token_embed == nullptr || token_embed->get_buffer() == nullptr) {
            tff::log::Logger::error("embedding kernel inputs is empty");
            return;
        }
        auto token_embed_buffer = token_embed->get_buffer();
        if (batch_token == nullptr || batch_token->get_buffer() == nullptr) {
            tff::log::Logger::error("embedding kernel inputs is empty");
            return;
        }
        auto batch_token_buffer = batch_token->get_buffer();
        auto dequantize_callback = core::memory::type_traits_auto[token_embed->get_data_type()].dequantize_callback;
        //
        auto &output_tensor = outputs;
        if (output_tensor == nullptr) {
            tff::log::Logger::error("embedding output tensor is empty");
            return;
        }
        Q8_0 *tmp_ptr = static_cast<Q8_0 *>(token_embed_buffer->ptr());
        for (size_t i = 0; i < batch_token->get_shape()[1]; ++i) {
            auto &shape_dim = batch_token->get_shape()[0];
            for (size_t j = 0; j < shape_dim; ++j) {
                int32_t token_id = batch_token->at<int32_t>(j);
                //tff::log::Logger::info("embedding token id=%d", token_id);
                auto quant_data_ptr = (const void *) (
                    (char *) token_embed_buffer->ptr() + token_id * token_embed->get_strides()[1]);

                auto float_data_ptr = (float *) (
                    (char *) output_tensor->get_buffer()->ptr() + (i * shape_dim * token_embed->get_shape()[0] + j *
                    token_embed->get_shape()[0]) * sizeof(float));
                if (float_data_ptr && quant_data_ptr) {
                    dequantize_callback(quant_data_ptr, float_data_ptr, token_embed->get_shape()[0]);
                }
            }
        }
    }

    //
    template<typename T>
    void tff::kernel::Embedding<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto embedding_weight = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            0, para_ptr);
        auto input_token = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        auto output_tensors = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);

        if (embedding_weight == nullptr || input_token == nullptr || output_tensors == nullptr) {
            tff::log::Logger::error("embedding input tensors is empty");
            return;
        }

        embedding_kernel_cpu<T>(embedding_weight, input_token, output_tensors);
    }

    template class tff::kernel::Embedding<float>;
    template class tff::kernel::Embedding<double>;
    template class tff::kernel::Embedding<int32_t>;
    template class tff::kernel::Embedding<Q8_0>;
    REGISTER_OP_OBJECT(Embedding, float);

    REGISTER_OP_OBJECT(Embedding, double);

    REGISTER_OP_OBJECT(Embedding, int32_t);

    REGISTER_OP_OBJECT(Embedding, Q8_0);
}
