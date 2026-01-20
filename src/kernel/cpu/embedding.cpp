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
                                     std::vector<std::shared_ptr<tff::core::memory::Tensor>> &inputs,
                                    std::shared_ptr<tff::core::memory::Tensor> &outputs,
                                     std::shared_ptr<
                                         tff::core::runtime::LLMMemManager> &mem_buffer_manager_ptr) {
        if (inputs.empty()) {
            tff::log::Logger::error("embedding kernel inputs is empty");
            return;
        }
        if (inputs.size() != 2) {
            tff::log::Logger::error("embedding kernel inputs size is invalid");
            return;
        }

        const auto& token_embed = *inputs.begin();
        if (token_embed == nullptr || token_embed->get_buffer() == nullptr) {
            tff::log::Logger::error("embedding kernel inputs is empty");
            return;
        }
        auto token_embed_buffer = token_embed->get_buffer();

        const auto& batch_token = *inputs.rbegin();
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
        //auto mem_buffer = mem_buffer_manager_ptr->get_cpu_mapped_memory();
        //output_tensor->set_buffer_data(mem_buffer.second, output_tensor->get_bytes(), mem_buffer.first);
        void *output_tensor_buffer;// = mem_buffer.second;

        for (size_t i = 0; i < batch_token->get_shape().size(); ++i) {
            auto &shape_dim = batch_token->get_shape()[i];
            for (size_t j = 0; j < shape_dim; ++j) {
                int32_t token_id = batch_token->at<int32_t>(j);
                tff::log::Logger::info("embedding token id=%d", token_id);
                auto quant_data_ptr = (const void *)((char *)token_embed_buffer->ptr() + token_id * token_embed->get_strides()[1]);
                auto float_data_ptr = (float *)((char *)output_tensor->get_buffer()->ptr()+ i * shape_dim + j);
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
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor>> >(
            1, para_ptr);
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        std::shared_ptr<core::runtime::LLMMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(3, para_ptr);

        if (input_tensors.empty()) {
            tff::log::Logger::error("embedding input tensors is empty");
            return;
        }
        if (output_tensors == nullptr) {
            tff::log::Logger::error("embedding output tensors is empty");
            return;
        }
        embedding_kernel_cpu<T>(input_tensors, output_tensors,
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
