//
// Created by nkk on 2025/11/10.
//

#include "QWen3Creator.h"

namespace tff::core::model {
    REGISTER_MODULE_OBJECT(QWen3Creator, ModelCreatorBase, MODEL_CREATOR_FLAG,
                           tff::core::model::to_string(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_QWEN3));

    //
    void QWen3Creator::build_graph(
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
            std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::model::layer::ModelLayerObject> > > > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
        //
        if (!graph_ptr) {
            graph_ptr = std::make_shared<tff::core::graph::Graph>();
        }

        const auto input_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT);
        if (input_layer_iter == layer_map.end()) {
            return;
        }
        const auto repeating_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
        if (repeating_layer_iter == layer_map.end()) {
            return;
        }
        //
        const auto output_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT);
        if (output_layer_iter == layer_map.end()) {
            return;
        }

        auto input_node = build_inputs(input_layer_iter->second.begin()->second);

        //
        if (input_layer_iter != layer_map.end() && !input_layer_iter->second.empty() &&
            repeating_layer_iter != layer_map.end() && !repeating_layer_iter->second.empty()) {
            //
            if (repeating_layer_iter != layer_map.end() && repeating_layer_iter->second.size() > 1) {
                const auto &repeating_layer_map = repeating_layer_iter->second; // Assume it's a std::set or similar
                for (size_t layer_id = 0; layer_id < repeating_layer_map.size(); ++layer_id) {
#ifdef _DEBUG
                    {
                        if (layer_id >= 1) {
                            continue;
                        }
                    }
#endif
                    tff::log::Logger::info("build layer :%d graph\n", layer_id);
                    auto &layer_map = repeating_layer_map.find(layer_id)->second;
                    //
                    auto attn_norm_node = build_norm(memory::ModelTensorType::LLM_TENSOR_ATTN_NORM,
                                                     layer_map, input_node);
                    //process qkv weight
                    {
                        auto attn_q_node = build_qkv_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q,
                                                          layer_map, attn_norm_node);

                        auto attn_k_node = build_qkv_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                                                          layer_map, attn_norm_node);
                        auto attn_v_node = build_qkv_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V,
                                                          layer_map, attn_norm_node);

                        auto attn_q_reshape_node = build_reshape_node(memory::ModelTensorType::LLM_TENSOR_ATTN_Q,
                                                                      layer_map, attn_q_node,
                                                                      this->_model_ctx._n_embd_head,
                                                                      this->_model_ctx._n_head,
                                                                      this->_model_ctx._n_tokens);
                        auto attn_k_reshape_node = build_reshape_node(memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                                                                      layer_map, attn_k_node,
                                                                      this->_model_ctx._n_embd_head,
                                                                      this->_model_ctx._n_head_kv,
                                                                      this->_model_ctx._n_tokens);
                        auto attn_v_reshape_node = build_reshape_node(memory::ModelTensorType::LLM_TENSOR_ATTN_V,
                                                                      layer_map, attn_v_node,
                                                                      this->_model_ctx._n_embd_head,
                                                                      this->_model_ctx._n_head_kv,
                                                                      this->_model_ctx._n_tokens);

                        auto attn_q_norm_node = build_norm(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q_NORM,
                                                           layer_map, attn_q_reshape_node);

                        auto attn_k_norm_node = build_norm(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K_NORM,
                                                           layer_map, attn_k_reshape_node);
                        //

                        auto attn_kv_store_cache_node = build_kv_cache_store_node(
                            layer_id, attn_k_norm_node, attn_v_reshape_node);
                        //
                        auto attn_k_load_node = build_kv_cache_load_node(memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                                                                         layer_id, attn_kv_store_cache_node);
                        //
                        auto attn_v_load_node = build_kv_cache_load_node(memory::ModelTensorType::LLM_TENSOR_ATTN_V,
                                                                         layer_id, attn_kv_store_cache_node);

                        //
                        auto rope_table_node = build_rope_table_node();
                        auto q_rope_node = build_rope_node(attn_q_norm_node, rope_table_node);
                        auto k_rope_node = build_rope_node(attn_k_load_node, rope_table_node);

                        auto attn_node = build_attn(layer_map, q_rope_node, k_rope_node, attn_v_load_node);
                        //
                        auto ffn_inp_node = build_ffn_inp(input_node, attn_node);

                        auto ffn_norm_node = build_norm(memory::ModelTensorType::LLM_TENSOR_FFN_NORM,
                                                        layer_map, ffn_inp_node);

                        //
                        auto ffn_node = build_ffn(layer_map, graph_ptr, ffn_norm_node);

                        auto result_node = build_add_node(ffn_inp_node, ffn_node);
                        NodeMetadata meta_result_node{ffn_node->name() + "_add_ffn_inp"};
                        result_node->set_node_meta(meta_result_node);
                        input_node = result_node;
                    }
                }
            }
        }

        auto output_norm_node = build_norm(memory::ModelTensorType::LLM_TENSOR_OUTPUT_NORM,
                                           output_layer_iter->second.begin()->second, input_node);
        auto out_put_node = build_output(memory::ModelTensorType::LLM_TENSOR_OUTPUT,
                                         output_layer_iter->second.begin()->second, graph_ptr,
                                         output_norm_node);

        graph_ptr->build_graph(out_put_node);
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_rope_table_node() {
        auto rope_table_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_PRE_ROPE_TABLE);
        if (rope_table_node != nullptr) {
            NodeMetadata meta_rope_node{false, false, std::string("pre_compute_rope_table_node")};
            rope_table_node->set_node_meta(meta_rope_node);
            auto para_ptr = rope_table_node->get_params();
            auto max_seq_len = this->_model_ctx._max_seq_len;
            auto embedding_dim = this->_model_ctx._n_embd_head;
            auto rope_base = this->_model_ctx._rope_freq_base;
            auto rope_scale = this->_model_ctx._rope_freq_scale;
            para_ptr->set_param<const uint32_t>(static_cast<const unsigned &&>(max_seq_len));
            para_ptr->set_param<const uint32_t>(static_cast<const unsigned &&>(embedding_dim));
            para_ptr->set_param<float>(std::move(rope_base));
            para_ptr->set_param<float>(std::move(rope_scale));


            auto dev_gpu = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
                DEVICE_BACKEND_FLAG,
                tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));

            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > devices = {{0, dev_gpu}};
            rope_table_node->bind_devices(devices);
            rope_table_node->set_tensor(std::make_shared<tff::core::memory::Tensor>(
                4, memory::DataType::TFF_DATA_TYPE_F32,
                memory::MemoryType::TFF_MEM_TYPE_WEIGHT,
                std::array<int64_t, MAX_TENSOR_DIM>{
                    embedding_dim, max_seq_len, 1, 1
                }));
            rope_table_node->get_tensor()->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_ROPE_FREQS);
        }
        return rope_table_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_mul_node(
        std::shared_ptr<core::model::layer::ModelLayerObject> &layer,
        std::shared_ptr<tff::core::graph::GraphNode> &a_node) {
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MUL);
        out_put_node->set_node_meta(NodeMetadata({a_node->name() + "_mul_w"}));
        auto devices = a_node->device();
        out_put_node->bind_devices(devices);
        out_put_node->add_input_node(a_node);

        auto weight_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_REF);
        const tff::core::graph::NodeMetadata meta_w_node{layer->_layer_name + "_w_ref"};
        weight_node->set_node_meta(meta_w_node);
        weight_node->bind_devices(layer->_device_list);
        weight_node->set_tensor(layer->_tensor);
        out_put_node->add_input_node(weight_node);

        auto tensor = std::make_shared<memory::Tensor>(a_node->get_tensor());
        tensor->set_tensor_type(a_node->get_tensor()->get_tensor_type());
        out_put_node->set_tensor(tensor);
        return out_put_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_mul_mat_node(
        std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> &layer,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node) {
        tff::core::graph::TffOpType op_type;
        auto weight_data_type = layer->_tensor->get_data_type();
        switch (weight_data_type) {
            case tff::core::memory::DataType::TFF_DATA_TYPE_Q8_0:
            case memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED:
                op_type = tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_MATMUL;
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F32:
            case tff::core::memory::DataType::TFF_DATA_TYPE_F64:
            case tff::core::memory::DataType::TFF_DATA_TYPE_F16:
            default:
                op_type = tff::core::graph::TffOpType::TFF_OP_MUL_MAT;
                break;
        }

        auto a_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_REF);
        a_node->set_node_meta(NodeMetadata(layer->_layer_name + "_w_ref"));
        a_node->bind_devices(layer->_device_list);
        a_node->set_tensor(layer->_tensor);

        auto result = ADD_NODE(op_type);
        tff::core::graph::NodeMetadata meta_qkv_w_mul_node{layer->_layer_name + "_mul_w"};
        result->set_node_meta(meta_qkv_w_mul_node);
        result->bind_devices(layer->_device_list);

        std::array<int64_t, MAX_TENSOR_DIM> shape = {
            layer->_tensor->get_shape()[1], b_node->get_tensor()->get_shape()[1], 1, 1
        };
        auto result_data_type = memory::DataType::TFF_DATA_TYPE_F32;
        switch (op_type) {
            case tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_MATMUL:
                result_data_type = memory::DataType::TFF_DATA_TYPE_F32;
                break;
            case tff::core::graph::TffOpType::TFF_OP_MUL_MAT:
            default:
                result_data_type = memory::DataType::TFF_DATA_TYPE_F32;
                break;
        }
        result->set_tensor(std::make_shared<tff::core::memory::Tensor>(layer->_tensor->get_shape().size(),
                                                                       result_data_type,
                                                                       memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                       shape));
        result->get_tensor()->set_tensor_type(layer->_tensor->get_tensor_type());

        result->add_input_node(a_node);
        if ((weight_data_type == tff::core::memory::DataType::TFF_DATA_TYPE_Q8_0 ||
             weight_data_type == tff::core::memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED) &&
            b_node->get_tensor()->get_data_type() == tff::core::memory::DataType::TFF_DATA_TYPE_F32) {
            auto quantize_node = ADD_NODE(TFF_OP_QUANTIZE_Q8);
            quantize_node->set_node_meta(NodeMetadata{"quantize_q_8_0_node"});
            quantize_node->bind_devices(layer->_device_list);
            quantize_node->add_input_node(b_node);
            std::array<int64_t, MAX_TENSOR_DIM> shape = {
                b_node->get_tensor()->get_shape()[0], b_node->get_tensor()->get_shape()[1], 1, 1
            };
            auto tensor = std::make_shared<tff::core::memory::Tensor>(memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED,
                                                                      memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                      shape);
            quantize_node->set_tensor(tensor);
            result->add_input_node(quantize_node);
        } else {
            result->add_input_node(b_node);
        }

        return result;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_add_node(
        std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node, bool inplace) {
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ADD);
        auto devices = a_node->device();
        out_put_node->bind_devices(devices);
        out_put_node->add_input_node(a_node);
        out_put_node->add_input_node(b_node);
        out_put_node->set_tensor(std::make_shared<tff::core::memory::Tensor>(a_node->get_tensor(), inplace));
        return out_put_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_norm(
        tff::core::memory::ModelTensorType tensor_type,
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        auto iter = layer_map.find(tensor_type);
        if (iter == layer_map.end()) {
            return nullptr;
        }
        auto layer = iter->second;

        auto rms_norm_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
        rms_norm_node->bind_devices(layer->_device_list);
        rms_norm_node->set_node_meta(NodeMetadata{layer->_layer_name + "_rms_norm"});
        rms_norm_node->get_params()->set_param(this->_model_ctx._f_norm_rms_eps);
        input_node = rms_norm_node->add_input_node(input_node);

        auto &inp_tensor = input_node->get_tensor();
        rms_norm_node->set_tensor(std::make_shared<memory::Tensor>(inp_tensor->get_shape().size(),
                                                                   inp_tensor->get_data_type(),
                                                                   memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                   inp_tensor->get_shape()));
        rms_norm_node->get_tensor()->set_tensor_type(inp_tensor->get_tensor_type());

        if (layer != nullptr) {
            auto rms_norm_mul_w_node = build_mul_node(layer, rms_norm_node);
            return rms_norm_mul_w_node;
        } else {
            return nullptr;
        }
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_rope_node(
        std::shared_ptr<tff::core::graph::GraphNode> &input_node,
        std::shared_ptr<tff::core::graph::GraphNode> &rope_table_node) {
        auto rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
        auto devices = input_node->device();
        rope_node->bind_devices(devices);
        NodeMetadata meta_q_rope_node{input_node->name() + "_rope"};
        rope_node->set_node_meta(meta_q_rope_node);
        rope_node->add_input_node(input_node);
        rope_node->add_input_node(rope_table_node);
        auto tensor = std::make_shared<memory::Tensor>(input_node->get_tensor()->get_shape().size(),
                                                       memory::DataType::TFF_DATA_TYPE_F32,
                                                       memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                       input_node->get_tensor()->get_shape());
        tensor->set_tensor_type(input_node->get_tensor()->get_tensor_type());
        rope_node->set_tensor(tensor);

        return rope_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_inputs(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD)->second;
        if (layer == nullptr) {
            return nullptr;
        }
        auto embd_weight_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_REF);
        const tff::core::graph::NodeMetadata meta_embd_w_node{true, false, layer->_layer_name + "_embedding"};
        embd_weight_node->set_node_meta(meta_embd_w_node);
        embd_weight_node->bind_devices(layer->_device_list);
        embd_weight_node->set_tensor(layer->_tensor);

        if (layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN) == layer_map.end()) {
            tff::log::Logger::error("current batch has no valid input token layer");
            return nullptr;
        }
        auto input_token_layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN)->second;
        this->_model_ctx._n_tokens = input_token_layer->_tensor->get_shape()[0];

        auto input_token_node = ADD_NODE(TFF_OP_MEM_REF);
        input_token_node->set_node_meta(NodeMetadata{true, false, input_token_layer->_layer_name});
        input_token_node->bind_devices(input_token_layer->_device_list);
        input_token_node->set_tensor(input_token_layer->_tensor);


        auto embedding_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_EMBEDDING);
        const tff::core::graph::NodeMetadata meta_tokenize_node{false, false, layer->_layer_name + "_embedding"};
        embedding_node->set_node_meta(meta_tokenize_node);
        embedding_node->bind_devices(layer->_device_list);
        embedding_node->add_input_node(embd_weight_node);
        embedding_node->add_input_node(input_token_node);

        std::array<int64_t, MAX_TENSOR_DIM> shape = {
            embd_weight_node->get_tensor()->get_shape()[0], input_token_node->get_tensor()->get_shape()[0],
            input_token_node->get_tensor()->get_shape()[1], 1
        };
        embedding_node->set_tensor(std::make_shared<tff::core::memory::Tensor>(
            MAX_TENSOR_DIM, memory::DataType::TFF_DATA_TYPE_F32,
            memory::MemoryType::TFF_MEM_TYPE_WORKSPACE, shape));

        return embedding_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_output(
        tff::core::memory::ModelTensorType tensor_type,
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        auto layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_OUTPUT)->second;
        auto result = build_mul_mat_node(layer, input_node);

        return result;
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_reshape_node(
        tff::core::memory::ModelTensorType tensor_type,
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node,
        int dim0, int dim1, int dim2) {
        auto attn_layer = layer_map.find(
                    tensor_type)->
                second;

        auto reshape_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
        tff::core::graph::NodeMetadata meta_qkv_reshape{input_node->name() + "_reshape"};
        reshape_node->set_node_meta(meta_qkv_reshape);
        auto device = input_node->device();
        reshape_node->bind_devices(device);

        std::array<int64_t, MAX_TENSOR_DIM> shapes = {dim0, dim1, dim2, 1};
        auto tensor = std::make_shared<memory::Tensor>(MAX_TENSOR_DIM, input_node->get_tensor()->get_data_type(),
                                                       memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                       shapes);
        tensor->set_tensor_type(input_node->get_tensor()->get_tensor_type());
        reshape_node->set_tensor(tensor);
        reshape_node->add_input_node(input_node);

        return reshape_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_qkv_node(
        tff::core::memory::ModelTensorType tensor_type,
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        auto attn_layer = layer_map.find(
                    tensor_type)->
                second;

        auto qkv_node = build_mul_mat_node(attn_layer,
                                           input_node);
        return qkv_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_attn(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::GraphNode> &q_node,
        std::shared_ptr<tff::core::graph::GraphNode> &k_node,
        std::shared_ptr<tff::core::graph::GraphNode> &v_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_OUT)->second;

        auto data_type = this->_model_ctx._use_fp16
                             ? memory::DataType::TFF_DATA_TYPE_F16
                             : memory::DataType::TFF_DATA_TYPE_F32;

        auto attn_mask_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ATTN_MASK);
        tff::core::graph::NodeMetadata meta_attn_mask{layer->_layer_name + "_attn_mask"};
        attn_mask_node->set_node_meta(meta_attn_mask);
        attn_mask_node->bind_devices(layer->_device_list);
        attn_mask_node->get_params()->set_param<const int>(core::graph::TFFMaskType::TFF_MASK_TYPE_CAUSAL);
        std::array<int64_t, MAX_TENSOR_DIM> shapes = {
            q_node->get_tensor()->get_shape()[1], q_node->get_tensor()->get_shape()[1], 1, 1
        };
        auto mask_tensor = std::make_shared<memory::Tensor>(data_type, memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
            shapes);
        mask_tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_ATTN_OUT);
        attn_mask_node->set_tensor(mask_tensor);

        auto flash_attn_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT);
        tff::core::graph::NodeMetadata meta_flash_attn_node{layer->_layer_name + "_flash_attn"};
        flash_attn_node->set_node_meta(meta_flash_attn_node);
        flash_attn_node->get_params()->set_param(this->_model_ctx._max_seq_len);
        flash_attn_node->bind_devices(layer->_device_list);
        flash_attn_node->add_input_node(q_node);
        flash_attn_node->add_input_node(k_node);
        flash_attn_node->add_input_node(v_node);
        flash_attn_node->add_input_node(attn_mask_node);

        auto tensor = std::make_shared<memory::Tensor>(data_type,memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                       std::array<int64_t, MAX_TENSOR_DIM>{
                                                           q_node->get_tensor()->get_shape()[0],
                                                           q_node->get_tensor()->get_shape()[1],
                                                           q_node->get_tensor()->get_shape()[2],
                                                           q_node->get_tensor()->get_shape()[3]
                                                       });
        tensor->set_tensor_type(memory::ModelTensorType::LLM_TENSOR_ATTN_OUT);
        flash_attn_node->set_tensor(tensor);

        auto flash_attn_mul_w_node = build_mul_mat_node(layer, flash_attn_node);

        return flash_attn_mul_w_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_ffn_inp(
        std::shared_ptr<tff::core::graph::GraphNode> &input_node,
        std::shared_ptr<tff::core::graph::GraphNode> &current_node) {
        auto ffn_inp = build_add_node(input_node, current_node);
        NodeMetadata meta_ffn_inp_node{current_node->name() + "_add_inp"};
        ffn_inp->set_node_meta(meta_ffn_inp_node);
        return ffn_inp;
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_ffn_up(
        const std::unordered_map<tff::core::memory::ModelTensorType,
            std::shared_ptr<tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_UP)->second;
        auto ffn_up_node = build_mul_mat_node(layer, input_node);
        return ffn_up_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_ffn_gate(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_GATE)->second;
        auto ffn_gate = build_mul_mat_node(layer, input_node);
        return ffn_gate;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_ffn_down(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_DOWN)->second;
        auto ffn_down_node = build_mul_mat_node(layer, input_node);
        return ffn_down_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_ffn(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        auto ffn_up_node = build_ffn_up(layer_map, graph_ptr, input_node);

        auto ffn_gate_node = build_ffn_gate(layer_map, graph_ptr, input_node);

        //silu;
        auto unary_op_node = build_unary_op(ffn_up_node, ffn_gate_node);

        return build_ffn_down(layer_map, graph_ptr, unary_op_node);
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_unary_op(
        std::shared_ptr<tff::core::graph::GraphNode> &up_node,
        std::shared_ptr<tff::core::graph::GraphNode> &gate_node) {
        auto unary_op_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_UNARY);
        auto devices = up_node->device();
        unary_op_node->bind_devices(devices);
        unary_op_node->set_node_meta({"swiglu_unary_op"});
        unary_op_node->add_input_node(gate_node);
        unary_op_node->add_input_node(up_node);

        unary_op_node->set_tensor(gate_node->get_tensor());
        unary_op_node->get_params()->set_param<const int>(tff::core::graph::TFFUnaryType::TFF_UNARY_TYPE_SILU);
        return unary_op_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_kv_cache_store_node(
        const int &layer_id,
        const std::shared_ptr<GraphNode> &k_node,
        const std::shared_ptr<GraphNode> &v_node) {
        auto cache_store_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_SET_ROWS);
        cache_store_node->set_node_meta({"cache_store_node"});

        auto device = k_node->device();
        cache_store_node->bind_devices(device);
        cache_store_node->add_input_node(k_node);
        cache_store_node->add_input_node(v_node);

        auto &input_tensor = k_node->get_tensor();
        auto device_iter = device.begin();
        cache_store_node->get_params()->set_param(this->_model_ctx._kv_cache_ptr[device_iter->first]);
        this->_model_ctx._kv_idx = this->_model_ctx._kv_cache_ptr[device_iter->first]->set_k(
            this->_model_ctx._seq_id, layer_id, input_tensor, device);
        cache_store_node->get_params()->set_param(this->_model_ctx._kv_idx);
        cache_store_node->get_params()->set_param(this->_model_ctx._seq_id);
        cache_store_node->get_params()->set_param(layer_id);
        cache_store_node->set_tensor(input_tensor);

        return cache_store_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_kv_cache_load_node(
        tff::core::memory::ModelTensorType tensor_type,
        const int &layer_id,
        const std::shared_ptr<GraphNode> &node) {
        auto cache_load_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_GET_ROWS);
        if (tensor_type == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
            cache_load_node->set_node_meta({"k_cache_load_node"});
        } else if (tensor_type == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
            cache_load_node->set_node_meta({"v_cache_load_node"});
        }
        auto device = node->device();
        auto device_iter = device.begin();
        cache_load_node->bind_devices(device);
        cache_load_node->add_input_node(node);

        auto &input_tensor = node->get_tensor();

        cache_load_node->get_params()->set_param(this->_model_ctx._kv_cache_ptr[device_iter->first]);
        this->_model_ctx._kv_idx = this->_model_ctx._kv_cache_ptr[device_iter->first]->set_k(
            this->_model_ctx._seq_id, layer_id, input_tensor, device);

        cache_load_node->get_params()->set_param(this->_model_ctx._kv_idx);
        cache_load_node->get_params()->set_param(this->_model_ctx._seq_id);
        cache_load_node->get_params()->set_param(layer_id);


        std::array<int64_t, MAX_TENSOR_DIM> shape = {
            input_tensor->get_shape()[0],
            input_tensor->get_shape()[1], input_tensor->get_shape()[2], input_tensor->get_shape()[3]
        };
        auto data_type = this->_model_ctx._use_fp16
                             ? memory::DataType::TFF_DATA_TYPE_F16
                             : memory::DataType::TFF_DATA_TYPE_F32;
        auto tensor = std::make_shared<tff::core::memory::Tensor>(data_type, memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
            shape);
        tensor->set_tensor_type(tensor_type);
        cache_load_node->set_tensor(tensor);

        return cache_load_node;
    }
}
