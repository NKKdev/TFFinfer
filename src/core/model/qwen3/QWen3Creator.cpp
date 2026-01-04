//
// Created by nkk on 2025/11/10.
//

#include "QWen3Creator.h"

namespace tff::core::model {
    REGISTER_MODULE_OBJECT(QWen3Creator, ModelCreatorBase, MODEL_CREATOR_FLAG,
                           tff::core::model::to_string(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_QWEN3));
#ifdef _EXPLICIT_DAG
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
        const auto repeating_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
        //
        const auto output_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT);

        NodeType input_node;
        build_inputs(input_layer_iter->second.begin()->second, graph_ptr, input_node);

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
                    NodeType attn_norm_node;
                    build_norm(memory::ModelTensorType::LLM_TENSOR_ATTN_NORM,
                        layer_map, graph_ptr, input_node, attn_norm_node);
                    //process qkv weight
                    {
                        NodeType attn_q_node;
                        build_qkv_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q,
                                       layer_map, graph_ptr, attn_norm_node, attn_q_node);
                        NodeType attn_k_node;
                        build_qkv_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                                       layer_map, graph_ptr, attn_norm_node, attn_k_node);
                        NodeType attn_v_node;
                        build_qkv_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V,
                                       layer_map, graph_ptr, attn_norm_node, attn_v_node);

                        NodeType attn_q_norm_node;
                        build_norm(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q,
                            layer_map, graph_ptr, attn_q_node, attn_q_norm_node);

                        NodeType attn_k_norm_node;
                        build_norm(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                            layer_map, graph_ptr, attn_k_node, attn_k_norm_node);
                        //


                        //
                        NodeType q_rope_node;
                        build_rope_node(layer_map, graph_ptr, attn_q_norm_node, q_rope_node);
                        NodeType k_rope_node;
                        build_rope_node(layer_map, graph_ptr, attn_k_norm_node, k_rope_node);

                        NodeType attn_node;
                        build_attn(layer_map, graph_ptr, input_node, q_rope_node, k_rope_node, attn_v_node,
                                   attn_node);
                        //
                        NodeType ffn_inp_node;
                        build_ffn_inp(layer_map, graph_ptr, input_node, attn_node, ffn_inp_node);

                        NodeType ffn_norm_node;
                        build_norm(memory::ModelTensorType::LLM_TENSOR_FFN_NORM,
                            layer_map, graph_ptr, ffn_inp_node, ffn_norm_node);

                        //
                        NodeType ffn_node;
                        build_ffn(layer_map, graph_ptr, ffn_norm_node, ffn_node);

                        auto result_node = build_add_node(
                            graph_ptr, ffn_inp_node.find(TFF_GRAPH_NODE_COMPUTE)->second,
                            ffn_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
                        NodeMetadata meta_result_node{
                            ffn_node.find(TFF_GRAPH_NODE_COMPUTE)->second->name() + "_add_ffn_inp_node"
                        };
                        result_node->set_node_meta(meta_result_node);

                        input_node[TFF_GRAPH_NODE_COMPUTE] = result_node;
                    }
                }
            }
        }

        NodeType output_norm_node;
        build_norm(memory::ModelTensorType::LLM_TENSOR_OUTPUT_NORM,
            output_layer_iter->second.begin()->second, graph_ptr, input_node, output_norm_node);
        NodeType out_put_node;
        build_output(memory::ModelTensorType::LLM_TENSOR_OUTPUT, output_layer_iter->second.begin()->second, graph_ptr, output_norm_node,
            out_put_node);

    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_rope_table_node(
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
        auto rope_table_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_PRE_ROPE_TABLE);
        if (rope_table_node != nullptr) {
            NodeMetadata meta_rope_node{std::string("pre_compute_rope_table_node")};
            rope_table_node->set_node_meta(meta_rope_node);
            auto para_ptr = rope_table_node->get_params();
            auto max_seq_len = _model_loader->get_model_config()._n_ctx;
            auto embedding_dim = _model_loader->get_model_config()._n_rot;
            para_ptr->set_param<const uint32_t>(para_ptr->get_param_count(),
                                                static_cast<const unsigned &&>(max_seq_len));
            para_ptr->set_param<const uint32_t>(para_ptr->get_param_count(),
                                                static_cast<const unsigned &&>(embedding_dim));
            auto dev_gpu = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
                DEVICE_BACKEND_FLAG,
                tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > devices = {{0, dev_gpu}};
            rope_table_node->bind_devices(devices);

            graph_ptr->add_node(rope_table_node);
        }
        return rope_table_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_mul_node(
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node) {
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MUL);
        auto devices = a_node->devices();
        out_put_node->bind_devices(devices);
        graph_ptr->add_node(out_put_node);
        graph_ptr->add_edge(a_node, out_put_node);
        graph_ptr->add_edge(b_node, out_put_node);
        return out_put_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_mul_mat_node(
        std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> &layer,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &a_node) {
        tff::core::graph::TffOpType op_type;
        switch (layer->_tensor->get_data_type()) {
            case tff::core::memory::DataType::TFF_DATA_TYPE_Q8_0:
                op_type = tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_MATMUL;
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F32:
            case tff::core::memory::DataType::TFF_DATA_TYPE_F64:
            case tff::core::memory::DataType::TFF_DATA_TYPE_F16:
            default:
                op_type = tff::core::graph::TffOpType::TFF_OP_MUL_MAT;
                break;
        }
        auto out_put_node = ADD_NODE(op_type);
        tff::core::graph::NodeMetadata meta_qkv_w_mul_node{layer->_layer_name + "_mul_w_node"};
        out_put_node->set_node_meta(meta_qkv_w_mul_node);

        out_put_node->bind_devices(layer->_device_list);
        out_put_node->add_inputs({layer->_tensor});
        graph_ptr->add_node(out_put_node);
        graph_ptr->add_edge(a_node, out_put_node);
        return out_put_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_add_node(
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node) {
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ADD);
        auto devices = a_node->device();
        out_put_node->bind_devices(devices);
        graph_ptr->add_node(out_put_node);
        graph_ptr->add_edge(a_node, out_put_node);
        graph_ptr->add_edge(b_node, out_put_node);
        return out_put_node;
    }
#ifdef _LOAD_WEIGHT_INFER
    //
    void QWen3Creator::build_cpu_node(
        std::shared_ptr<GraphNode> &layer,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        NodeType &out_put_node) {
        //auto current_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
        graph_ptr->add_node(layer);
        auto device = tff::factory::ModuleFactory::instance()->create_shared<
            tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));

        layer->bind_devices(device);
        tff::core::graph::NodeMetadata meta_map2cpu{layer->name() + "_map2cpu"};
        layer->set_node_meta(meta_map2cpu);

        if (this->_current_mem_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU) == this->
            _current_mem_node.end()) {
            this->_current_mem_node[tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU] = layer;
        } else {
            graph_ptr->add_edge(
                this->_current_mem_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU)->second,
                layer);
        }

        out_put_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU, layer});
        //
        update_mem_node(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU, layer);
    }

    //
    void QWen3Creator::build_gpu_node(
        const std::string &node_name,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        NodeType &out_put_node, bool is_input) {
        auto current_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
        graph_ptr->add_node(current_cpu2gpu_node);
        tff::core::graph::NodeMetadata meta_cpu2gpu{node_name + "_cpu2gpu"};
        current_cpu2gpu_node->set_node_meta(meta_cpu2gpu);
        auto dev_gpu = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
        current_cpu2gpu_node->bind_devices(dev_gpu);
        current_cpu2gpu_node->get_params()->set_param(1, tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE);
        if (!is_input) {
            graph_ptr->add_edge(
                this->_current_mem_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU)->second,
                current_cpu2gpu_node);
        }


        if (this->_current_mem_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU) == this->
            _current_mem_node.end()) {
            this->_current_mem_node[tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU] = current_cpu2gpu_node;
        }
        //else {
        //     graph_ptr->add_edge(
        //         this->_current_mem_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU)->second,
        //         current_cpu2gpu_node);
        // }
        out_put_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU, current_cpu2gpu_node});

        //
        update_mem_node(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU, current_cpu2gpu_node);
    }
#endif
    //
    void QWen3Creator::build_norm(tff::core::memory::ModelTensorType tensor_type,
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                           tff::core::model::layer::ModelLayerObject> > &layer_map,
                                       std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                       NodeType &input_node,
                                       NodeType &norm_node) {
        auto layer = layer_map.find(tensor_type)->second;
#ifdef _LOAD_WEIGHT_INFER
        build_cpu_node(layer, graph_ptr, attn_norm_node);
        build_gpu_node(layer->_layer_name, graph_ptr, attn_norm_node);
#endif

        //
        auto rms_norm_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
        rms_norm_node->set_node_meta(NodeMetadata{layer->_layer_name + "_rms_norm_node"});
        rms_norm_node->bind_devices(layer->_device_list);
        rms_norm_node->add_inputs({layer->_tensor});

        graph_ptr->add_node(rms_norm_node);
        graph_ptr->add_edge(input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second,
                            rms_norm_node);
#ifdef _LOAD_WEIGHT_INFER
        graph_ptr->add_edge(attn_norm_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
                            rms_norm_node);
#endif

        norm_node.insert({TFF_GRAPH_NODE_COMPUTE, rms_norm_node});
    }
    void QWen3Creator::build_rope_node(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                           tff::core::model::layer::ModelLayerObject> > &layer_map,
                                       std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                       NodeType &input_node,
                                       NodeType &oput_put_node) {

        auto rope_table_node = build_rope_table_node(graph_ptr);

        auto rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
        auto devices = input_node[TFF_GRAPH_NODE_COMPUTE]->devices();
        rope_node->bind_devices(devices);

        NodeMetadata meta_q_rope_node{
            input_node.find(TFF_GRAPH_NODE_COMPUTE)->second->name() + "_rope"
        };
        rope_node->set_node_meta(meta_q_rope_node);
        graph_ptr->add_node(rope_node);
        graph_ptr->add_edge(input_node.find(TFF_GRAPH_NODE_COMPUTE)->second, rope_node);
        graph_ptr->add_edge(rope_table_node, rope_node);
        oput_put_node[TFF_GRAPH_NODE_COMPUTE] = rope_node;
    }
    //
    void QWen3Creator::build_inputs(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        NodeType &input_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD)->second;

        auto embedding_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_EMBEDDING);
        const tff::core::graph::NodeMetadata meta_tokenize_node{true, false, layer->_layer_name + "_embedding"};
        embedding_node->set_node_meta(meta_tokenize_node);
        embedding_node->bind_devices(layer->_device_list);
        embedding_node->add_inputs({layer->_tensor});
        graph_ptr->add_node(embedding_node);

        if (layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN) == layer_map.end()) {
            tff::log::Logger::error("current batch has no valid input token layer");
            return;
        }
        auto input_token_layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN)->second;
        auto input_token_node = ADD_NODE(TFF_OP_MEM_REF);
        input_token_node->set_node_meta(NodeMetadata{input_token_layer->_layer_name});
        input_token_node->bind_devices(input_token_layer->_device_list);
        input_token_node->add_inputs({input_token_layer->_tensor});
        graph_ptr->add_node(input_token_node);
        //graph_ptr->add_edge(input_node.find(TFF_GRAPH_NODE_MAP2CPU)->second, embedding_node);
        graph_ptr->add_edge(input_token_node, embedding_node);

        auto input_pos_layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_TOKEN_POS)->second;
        auto input_pos_node = ADD_NODE(TFF_OP_MEM_REF);
        input_pos_node->set_node_meta(NodeMetadata{input_pos_layer->_layer_name});
        input_pos_node->bind_devices(input_token_layer->_device_list);
        input_pos_node->add_inputs({input_pos_layer->_tensor});
        graph_ptr->add_node(input_pos_node);
        graph_ptr->add_edge(input_pos_node, embedding_node);


        input_node.insert({
            tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE, embedding_node
        });

    }
    //
    void QWen3Creator::build_output(tff::core::memory::ModelTensorType tensor_type,
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                           tff::core::model::layer::ModelLayerObject> > &layer_map,
                                       std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                       NodeType &input_node,
                                       NodeType &out_put_node) {
        auto layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_OUTPUT)->second;
        out_put_node[TFF_GRAPH_NODE_COMPUTE] = build_mul_mat_node(layer, graph_ptr, input_node[TFF_GRAPH_NODE_COMPUTE]);
    }
    //
    void QWen3Creator::build_qkv_node(tff::core::memory::ModelTensorType tensor_type,
                                      const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                          tff::core::model::layer::ModelLayerObject> > &layer_map,
                                      std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                      NodeType &input_node,
                                      NodeType &attn_qkv_node) {
        auto attn_layer = layer_map.find(
                    tensor_type)->
                second;
#ifdef _LOAD_WEIGHT_INFER
        build_cpu_node(layer, graph_ptr, attn_qkv_node);
        build_gpu_node(node_name, graph_ptr, attn_qkv_node);
#endif

        auto qkv_node = build_mul_mat_node(attn_layer,
                                           graph_ptr,
                                           input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)
                                           ->second);

        auto reshape_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
        tff::core::graph::NodeMetadata meta_qkv_reshape{attn_layer->_layer_name + "_reshape"};
        reshape_node->set_node_meta(meta_qkv_reshape);
        reshape_node->bind_devices(attn_layer->_device_list);
        graph_ptr->add_node(reshape_node);
        graph_ptr->add_edge(qkv_node,
                            reshape_node);

        attn_qkv_node.insert({TFF_GRAPH_NODE_COMPUTE, reshape_node});
    }

    void QWen3Creator::build_attn(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                      tff::core::model::layer::ModelLayerObject> > &layer_map,
                                  std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                  NodeType &input_node,
                                  NodeType &q_node,
                                  NodeType &k_node,
                                  NodeType &v_node,
                                  NodeType &out_put_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_OUT)->second;
#ifdef _LOAD_WEIGHT_INFER
        build_cpu_node(wo_layer, graph_ptr, out_put_node);
        build_gpu_node(node_name, graph_ptr, out_put_node);
#endif

        auto flash_attn_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT);
        tff::core::graph::NodeMetadata meta_flash_attn_node{layer->_layer_name + "_flash_attn_node"};
        flash_attn_node->set_node_meta(meta_flash_attn_node);
        flash_attn_node->bind_devices(layer->_device_list);

        graph_ptr->add_node(flash_attn_node);
        graph_ptr->add_edge(q_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
        graph_ptr->add_edge(k_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
        graph_ptr->add_edge(v_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);


        auto flash_attn_mul_w_node = build_mul_mat_node(layer, graph_ptr,
                                              flash_attn_node);


        out_put_node.insert({TFF_GRAPH_NODE_COMPUTE, flash_attn_mul_w_node});
    }
    //
    void QWen3Creator::build_ffn_inp(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                           tff::core::model::layer::ModelLayerObject> > &layer_map,
                                       std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                       NodeType &input_node,
                                       NodeType &current_node,
                                       NodeType &ffn_inp_node) {
        auto ffn_inp = build_add_node(graph_ptr, input_node.find(TFF_GRAPH_NODE_COMPUTE)->second,
                                      current_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
        NodeMetadata meta_ffn_inp_node{
            current_node.find(TFF_GRAPH_NODE_COMPUTE)->second->name() + "_add_inp"
        };
        ffn_inp->set_node_meta(meta_ffn_inp_node);
        ffn_inp_node[TFF_GRAPH_NODE_COMPUTE] = ffn_inp;
    }
    void QWen3Creator::build_ffn_up(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                        tff::core::model::layer::ModelLayerObject> > &layer_map,
                                    std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                    NodeType &input_node,
                                    NodeType &ffn_up_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_UP)->second;
#ifdef _LOAD_WEIGHT_INFER
        build_cpu_node(layer, graph_ptr, ffn_up_node);
        build_gpu_node(node_name, graph_ptr, ffn_up_node);
#endif

        ffn_up_node[TFF_GRAPH_NODE_COMPUTE] = build_mul_mat_node(layer, graph_ptr,
                                                input_node[TFF_GRAPH_NODE_COMPUTE]);

    }

    //
    void QWen3Creator::build_ffn_gate(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                          tff::core::model::layer::ModelLayerObject> > &layer_map,
                                      std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                      NodeType &input_node,
                                      NodeType &ffn_gate_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_GATE)->second;
#ifdef _LOAD_WEIGHT_INFER
        build_cpu_node(layer, graph_ptr, ffn_gate_node);
        build_gpu_node(node_name, graph_ptr, ffn_gate_node);
#endif
        ffn_gate_node[TFF_GRAPH_NODE_COMPUTE] = build_mul_mat_node(layer, graph_ptr,
                                                  input_node[TFF_GRAPH_NODE_COMPUTE]);

    }

    //
    void QWen3Creator::build_ffn_down(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                          tff::core::model::layer::ModelLayerObject> > &layer_map,
                                      std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                      NodeType &input_node,
                                      NodeType &ffn_down_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_DOWN)->second;
#ifdef _LOAD_WEIGHT_INFER
        build_cpu_node(layer, graph_ptr, ffn_doown_node);
        build_gpu_node(node_name, graph_ptr, ffn_down_node);
#endif
        ffn_down_node[TFF_GRAPH_NODE_COMPUTE] = build_mul_mat_node(layer, graph_ptr,
                                                  input_node[TFF_GRAPH_NODE_COMPUTE]);
    }

    //
    void QWen3Creator::build_ffn(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                     tff::core::model::layer::ModelLayerObject> > &layer_map,
                                 std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                 NodeType &input_node,
                                 NodeType &ffn_node) {
        NodeType ffn_up_node;
        build_ffn_up(layer_map, graph_ptr, input_node, ffn_up_node);

        NodeType ffn_gate_node;
        build_ffn_gate(layer_map, graph_ptr, ffn_up_node, ffn_gate_node);

        build_ffn_down(layer_map, graph_ptr, ffn_gate_node, ffn_node);

    }
#else

#endif
}
