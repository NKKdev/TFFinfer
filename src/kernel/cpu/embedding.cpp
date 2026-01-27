//
// Created by nkk on 2025/11/18.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMMemManager.h"
#include "kernel/include/BaseDefine.h"

namespace tff::kernel {
    template<typename T>
    static void embedding_kernel_cpu(
        std::shared_ptr<tff::core::memory::Tensor> &token_embed,
        std::shared_ptr<tff::core::memory::Tensor> &batch_token,
        std::shared_ptr<tff::core::memory::Tensor> &outputs,
        std::shared_ptr<
            tff::core::runtime::LLMMemManager> &mem_buffer_manager_ptr) {
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
        for (size_t i = 0; i < batch_token->get_shape()[1]; ++i) {
            auto &shape_dim = batch_token->get_shape()[0];
            for (size_t j = 0; j < shape_dim; ++j) {
                int32_t token_id = batch_token->at<int32_t>(j);
                tff::log::Logger::info("embedding token id=%d", token_id);
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
        tff::log::Logger::info("layer node op:%s compute!", tff::kernel::Embedding<T>::get_op_name().c_str());
        const auto &name = get_param_value<std::string>(0, para_ptr);
        auto embedding_weight = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        auto input_token = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            3, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(4, para_ptr);

        if (embedding_weight == nullptr || input_token == nullptr || output_tensors == nullptr) {
            tff::log::Logger::error("embedding input tensors is empty");
            return;
        }

        embedding_kernel_cpu<T>(embedding_weight, input_token, output_tensors,
                                mem_buffer_manager_ptr);
    }

    template<typename T>
    std::string tff::kernel::Embedding<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_EMBEDDING);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CPU + tff::core::global::get_type_suffix<T>();;
        return name;
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
