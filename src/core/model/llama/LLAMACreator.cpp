//
// Created by nkk on 2025/11/10.
//

#include "LLAMACreator.h"

namespace tff::core::model {
    REGISTER_MODULE_OBJECT(LLAMACreator, ModelCreatorBase, MODEL_CREATOR_FLAG,
                           std::string(LLM_ARCH_NAMES.find(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA
                           )->second));

    void LLAMACreator::build_layer(std::shared_ptr<tff::core::memory::Tensor> &tensor_ptr,
                                   std::shared_ptr<tff::core::graph::GraphNode> &layer_node,
                                   const size_t &total_layer_num, const size_t &layer_index) {
        auto &layer_info = LLM_LAYER_OP_INFOS.find(tensor_ptr->get_tensor_type())->second;
        layer_node = tff::factory::ModuleFactory::instance()->create_shared<tff::core::graph::GraphNode>(
            OP_NODE_FLAG, layer_info.second);
        if (!layer_node) {
            return;
        }
        // auto layer_params_ptr = std::make_shared<tff::core::global::ParamBaseObject>();
        // layer_node->set_params(layer_params_ptr);
        layer_node->set_layer_id(layer_index);
        layer_node->set_layer_type(layer_info.first);
        layer_node->set_inputs(
            std::vector<std::shared_ptr<tff::core::memory::Tensor> >{tensor_ptr});
        switch (layer_info.first) {
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT: {
                auto device = tff::factory::ModuleFactory::instance()->create_shared<
                    tff::core::device::DeviceBaseObject>(
                    DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
                layer_node->bind_devices(device);
                break;
            }
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT: {
                auto device_cuda = tff::factory::ModuleFactory::instance()->create_shared<
                    tff::core::device::DeviceBaseObject>(
                    DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
                layer_node->bind_devices(device_cuda);
                break;
            }
            case tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING: {
                switch (layer_info.second) {
                    case tff::core::graph::TffOpType::TFF_OP_MAP2CPU: {
                        auto device = tff::factory::ModuleFactory::instance()->create_shared<
                            tff::core::device::DeviceBaseObject>(
                            DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
                        layer_node->bind_devices(device);
                        break;
                    }
                    default: {
                        auto device_size = tff::core::device::get_device_size(DEVICE_BACKEND_TYPE_CUDA);
                        std::vector<float> device_splits;
                        auto device_cuda = tff::factory::ModuleFactory::instance()->create_shared<
                            tff::core::device::DeviceBaseObject>(
                            DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));


                        std::vector<int> device_list;
                        device_cuda->get_device_id(device_list);
                        for (size_t i = 0; i < device_list.size(); ++i) {
                            size_t total_mem;
                            size_t free_mem;
                            device_cuda->get_device_mem(i, &free_mem, &total_mem);
                            device_splits.push_back(static_cast<float>(free_mem));
                        }

                        //计算切分点;
                        float split_sum = 0.0f;
                        for (size_t i = 0; i < device_size; ++i) {
                            split_sum += device_splits[i];
                            device_splits[i] = split_sum;
                        }
                        for (size_t i = 0; i < device_size; ++i) {
                            device_splits[i] /= split_sum;
                        }
                        //
                        const int layer_gpu = std::upper_bound(device_splits.begin(),
                                                               device_splits.begin() + device_size,
                                                               float(layer_index) / total_layer_num) - device_splits.
                                              begin();
                        layer_node->bind_devices(device_cuda); //todo 应该绑定某种类型设备下某个设备
                    }
                }
                break;
            }
            default:
                break;
        }
    }

    //
    void LLAMACreator::build_graph(
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
            std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::graph::GraphNode> > > > &layer_map,
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
                    build_attn_norm(layer_map, graph_ptr, input_node, attn_norm_node);
                    //process qkv weight
                    {
                        auto attn_q_layer = layer_map.find(
                                    tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q)->
                                second;
                        NodeType attn_q_node;
                        build_qkv_node(attn_q_layer, graph_ptr, attn_norm_node, attn_q_node);

                        auto attn_k_layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K)->
                                second;
                        NodeType attn_k_node;
                        build_qkv_node(attn_k_layer, graph_ptr, attn_norm_node, attn_k_node);

                        auto attn_v_layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V)->
                                second;
                        NodeType attn_v_node;
                        build_qkv_node(attn_v_layer, graph_ptr, attn_norm_node, attn_v_node);
                        //
                        auto rope_table_node = build_rope_table_node(graph_ptr);

                        //
                        auto q_rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
                        q_rope_node->bind_devices(attn_q_node.find(TFF_GRAPH_NODE_COMPUTE)->second->device());
                        NodeMetadata meta_q_rope_node{
                            attn_q_node.find(TFF_GRAPH_NODE_COMPUTE)->second->name() + "_rope"
                        };
                        q_rope_node->set_node_meta(meta_q_rope_node);
                        graph_ptr->add_node(q_rope_node);
                        graph_ptr->add_edge(attn_q_node.find(TFF_GRAPH_NODE_COMPUTE)->second, q_rope_node);
                        graph_ptr->add_edge(rope_table_node, q_rope_node);
                        attn_q_node[TFF_GRAPH_NODE_COMPUTE] = q_rope_node;

                        //
                        auto k_rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
                        k_rope_node->bind_devices(attn_k_node.find(TFF_GRAPH_NODE_COMPUTE)->second->device());
                        NodeMetadata meta_k_rope_node{
                            attn_k_node.find(TFF_GRAPH_NODE_COMPUTE)->second->name() + "_rope"
                        };
                        k_rope_node->set_node_meta(meta_k_rope_node);
                        graph_ptr->add_node(k_rope_node);
                        graph_ptr->add_edge(attn_k_node.find(TFF_GRAPH_NODE_COMPUTE)->second, k_rope_node);
                        graph_ptr->add_edge(rope_table_node, k_rope_node);
                        attn_k_node[TFF_GRAPH_NODE_COMPUTE] = k_rope_node;


                        NodeType attn_node;
                        build_attn(layer_map, graph_ptr, input_node, attn_q_node, attn_k_node, attn_v_node,
                                   attn_node);
                        //
                        NodeType ffn_inp_node;
                        auto ffn_inp = build_add_node(graph_ptr, attn_node.find(TFF_GRAPH_NODE_COMPUTE)->second,
                                                      input_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
                        NodeMetadata meta_ffn_inp_node{
                            attn_node.find(TFF_GRAPH_NODE_COMPUTE)->second->name() + "_add_inp"
                        };
                        ffn_inp->set_node_meta(meta_ffn_inp_node);
                        ffn_inp_node[TFF_GRAPH_NODE_COMPUTE] = ffn_inp;
                        //
                        NodeType ffn_node;
                        build_ffn(layer_map, graph_ptr, ffn_inp_node, ffn_node);

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

        //NodeType output_norm_node;
        //build_output_norm(output_layer_iter->second.begin()->second, graph_ptr, input_node, output_norm_node);
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> LLAMACreator::build_rope_table_node(
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
        auto rope_table_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_PRE_ROPE_TABLE);
        if (rope_table_node != nullptr) {
            NodeMetadata meta_rope_node{std::string("pre_compute_rope_table_node")};
            rope_table_node->set_node_meta(meta_rope_node);
            auto para_ptr = rope_table_node->get_params();
            auto max_seq_len = _model_loader->get_model_config()._n_ctx;
            auto embedding_dim = _model_loader->get_model_config()._n_rot;
            para_ptr->set_param<const uint32_t>(para_ptr->get_param_count(), static_cast<const unsigned &&>(max_seq_len));
            para_ptr->set_param<const uint32_t>(para_ptr->get_param_count(), static_cast<const unsigned &&>(embedding_dim));
            auto dev_gpu = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
                DEVICE_BACKEND_FLAG,
                tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
            rope_table_node->bind_devices(dev_gpu);

            graph_ptr->add_node(rope_table_node);
        }
        return rope_table_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> LLAMACreator::build_mul_node(
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr, std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node) {
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MUL);
        out_put_node->bind_devices(a_node->device());
        graph_ptr->add_node(out_put_node);
        graph_ptr->add_edge(a_node, out_put_node);
        graph_ptr->add_edge(b_node, out_put_node);
        return out_put_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> LLAMACreator::build_mul_mat_node(
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node) {
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MUL_MAT);
        out_put_node->bind_devices(a_node->device());
        graph_ptr->add_node(out_put_node);
        graph_ptr->add_edge(a_node, out_put_node);
        graph_ptr->add_edge(b_node, out_put_node);
        return out_put_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> LLAMACreator::build_add_node(
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node) {
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ADD);
        out_put_node->bind_devices(a_node->device());
        graph_ptr->add_node(out_put_node);
        graph_ptr->add_edge(a_node, out_put_node);
        graph_ptr->add_edge(b_node, out_put_node);
        return out_put_node;
    }

    //
    void LLAMACreator::build_cpu_node(
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
    void LLAMACreator::build_gpu_node(
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

    //
    void LLAMACreator::build_attn_norm(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                           tff::core::graph::GraphNode> > &layer_map,
                                       std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                       NodeType &input_node,
                                       NodeType &attn_norm_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_NORM)->second;
        std::string node_name = layer->name();
        build_cpu_node(layer, graph_ptr, attn_norm_node);
        build_gpu_node(node_name, graph_ptr, attn_norm_node);


        //
        auto rms_norm_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
        rms_norm_node->set_node_meta(NodeMetadata{node_name + "_rms_norm_node"});
        rms_norm_node->bind_devices(attn_norm_node.find(TFF_GRAPH_NODE_CPU2GPU)->second->device());

        graph_ptr->add_node(rms_norm_node);
        graph_ptr->add_edge(input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second,
                            rms_norm_node);
        graph_ptr->add_edge(attn_norm_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
                            rms_norm_node);

        attn_norm_node.insert({TFF_GRAPH_NODE_COMPUTE, rms_norm_node});
    }

    //
    void LLAMACreator::build_inputs(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::graph::GraphNode> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        NodeType &input_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD)->second;
        std::string node_name = layer->name();
        build_cpu_node(layer, graph_ptr, input_node);
        build_gpu_node(node_name, graph_ptr, input_node, true);

        auto embedding_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_EMBEDDING);
        const tff::core::graph::NodeMetadata meta_tokenize_node{true, false, node_name + "_embedding"};
        embedding_node->set_node_meta(meta_tokenize_node);
        embedding_node->bind_devices(layer->device());
        graph_ptr->add_node(embedding_node);

        if (layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN) == layer_map.end()) {
            tff::log::Logger::error("current batch has no valid input token layer");
            return;
        }
        auto input_token_layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN)->second;
        input_token_layer->set_node_meta(NodeMetadata{node_name + "_input_token"});
        input_token_layer->bind_devices(layer->device());
        graph_ptr->add_node(input_token_layer);
        graph_ptr->add_edge(input_node.find(TFF_GRAPH_NODE_MAP2CPU)->second, embedding_node);
        graph_ptr->add_edge(input_token_layer, embedding_node);

        graph_ptr->add_edge(embedding_node,
                            input_node.find(TFF_GRAPH_NODE_CPU2GPU)->second);

        //
        input_node.insert({
            tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE, input_node.find(TFF_GRAPH_NODE_CPU2GPU)->second
        });
        //
        update_mem_node(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU,
                        input_node.find(TFF_GRAPH_NODE_MAP2CPU)->second);
        update_mem_node(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU,
                        input_node.find(TFF_GRAPH_NODE_CPU2GPU)->second);
    }

    //
    void LLAMACreator::build_qkv_node(std::shared_ptr<GraphNode> &layer,
                                      std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                      NodeType &input_node,
                                      NodeType &attn_qkv_node) {
        std::string node_name = layer->name();

        build_cpu_node(layer, graph_ptr, attn_qkv_node);
        build_gpu_node(node_name, graph_ptr, attn_qkv_node);

        auto qkv_node = build_mul_mat_node(graph_ptr,
                                           input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)
                                           ->second,
                                           attn_qkv_node.find(
                                               tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU)->second);
        tff::core::graph::NodeMetadata meta_qkv_w_mul_node{node_name + "_qkv_mul_w_node"};
        qkv_node->set_node_meta(meta_qkv_w_mul_node);
        auto para_ptr = qkv_node->get_params();
        para_ptr->set_param(para_ptr->get_param_count(), tff::core::graph::MatMulTransType::TFF_TT);


        auto reshape_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
        tff::core::graph::NodeMetadata meta_qkv_reshape{node_name + "_qkv_reshape"};
        reshape_node->set_node_meta(meta_qkv_reshape);
        reshape_node->bind_devices(qkv_node->device());
        graph_ptr->add_node(reshape_node);
        graph_ptr->add_edge(qkv_node,
                            reshape_node);

        attn_qkv_node.insert({TFF_GRAPH_NODE_COMPUTE, reshape_node});
    }

    //
    void LLAMACreator::build_attn(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                      tff::core::graph::GraphNode> > &layer_map,
                                  std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                  NodeType &input_node,
                                  NodeType &q_node,
                                  NodeType &k_node,
                                  NodeType &v_node,
                                  NodeType &out_put_node) {
        auto wo_layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_OUT)->second;
        std::string node_name = wo_layer->name();
        build_cpu_node(wo_layer, graph_ptr, out_put_node);
        build_gpu_node(node_name, graph_ptr, out_put_node);


        auto flash_attn_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT);
        tff::core::graph::NodeMetadata meta_flash_attn_node{node_name + "_flash_attn_node"};
        flash_attn_node->set_node_meta(meta_flash_attn_node);
        flash_attn_node->bind_devices(out_put_node.find(TFF_GRAPH_NODE_CPU2GPU)->second->device());

        graph_ptr->add_node(flash_attn_node);
        //graph_ptr->add_edge(input_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
        graph_ptr->add_edge(q_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
        graph_ptr->add_edge(k_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
        graph_ptr->add_edge(v_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);


        auto attn_norm_w = build_mul_mat_node(graph_ptr, out_put_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
                                              flash_attn_node);
        tff::core::graph::NodeMetadata meta_attn_norm_w{node_name + "_mul_w_node"};
        attn_norm_w->set_node_meta(meta_attn_norm_w);
        auto para_ptr = attn_norm_w->get_params();
        para_ptr->set_param(para_ptr->get_param_count(), tff::core::graph::MatMulTransType::TFF_TT);

        out_put_node.insert({TFF_GRAPH_NODE_COMPUTE, attn_norm_w});
    }

    void LLAMACreator::build_ffn_up(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                        tff::core::graph::GraphNode> > &layer_map,
                                    std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                    NodeType &input_node,
                                    NodeType &ffn_up_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_UP)->second;
        std::string node_name = layer->name();

        build_cpu_node(layer, graph_ptr, ffn_up_node);
        build_gpu_node(node_name, graph_ptr, ffn_up_node);


        auto ffn_up_w_node = build_mul_mat_node(graph_ptr, ffn_up_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
                                                input_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
        tff::core::graph::NodeMetadata meta_ffn_up_w_node{node_name + "_mul_w_node"};
        ffn_up_w_node->set_node_meta(meta_ffn_up_w_node);
        auto para_ptr = ffn_up_w_node->get_params();
        para_ptr->set_param(para_ptr->get_param_count(), tff::core::graph::MatMulTransType::TFF_TT);

        // auto ffn_up_b_node = build_add_node(graph_ptr, ffn_up_w_node,
        //                                     ffn_up_node.find(TFF_GRAPH_NODE_CPU2GPU)->second);
        // tff::core::graph::NodeMetadata meta_ffn_up_b_node{node_name + "_ffn_up_b_node"};
        // ffn_up_b_node->set_node_meta(meta_ffn_up_b_node);

        ffn_up_node[TFF_GRAPH_NODE_COMPUTE] = ffn_up_w_node;
    }

    //
    void LLAMACreator::build_ffn_gate(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                          tff::core::graph::GraphNode> > &layer_map,
                                      std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                      NodeType &input_node,
                                      NodeType &ffn_gate_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_GATE)->second;
        std::string node_name = layer->name();
        build_cpu_node(layer, graph_ptr, ffn_gate_node);
        build_gpu_node(node_name, graph_ptr, ffn_gate_node);

        auto ffn_gate_w_node = build_mul_mat_node(graph_ptr, ffn_gate_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
                                                  input_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
        tff::core::graph::NodeMetadata meta_ffn_gate_w_node{node_name + "_mul_w_node"};
        ffn_gate_w_node->set_node_meta(meta_ffn_gate_w_node);
        auto para_ptr = ffn_gate_w_node->get_params();
        para_ptr->set_param(para_ptr->get_param_count(), tff::core::graph::MatMulTransType::TFF_TT);

        // auto ffn_gate_b_node = build_add_node(graph_ptr, ffn_gate_w_node,
        //                                       ffn_gate_node.find(TFF_GRAPH_NODE_CPU2GPU)->second);
        // tff::core::graph::NodeMetadata meta_ffn_gate_b_node{node_name + "ffn_gate_b_node"};
        // ffn_gate_b_node->set_node_meta(meta_ffn_gate_b_node);

        ffn_gate_node[TFF_GRAPH_NODE_COMPUTE] = ffn_gate_w_node;
    }

    //
    void LLAMACreator::build_ffn_down(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                          tff::core::graph::GraphNode> > &layer_map,
                                      std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                      NodeType &input_node,
                                      NodeType &ffn_down_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_DOWN)->second;
        std::string node_name = layer->name();
        build_cpu_node(layer, graph_ptr, ffn_down_node);
        build_gpu_node(node_name, graph_ptr, ffn_down_node);

        auto ffn_down_w_node = build_mul_mat_node(graph_ptr, ffn_down_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
                                                  input_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
        tff::core::graph::NodeMetadata meta_ffn_down_w_node{node_name + "_mul_w_node"};
        ffn_down_w_node->set_node_meta(meta_ffn_down_w_node);
        auto para_ptr = ffn_down_w_node->get_params();
        para_ptr->set_param(para_ptr->get_param_count(), tff::core::graph::MatMulTransType::TFF_TT);

        // auto ffn_down_b_node = build_add_node(graph_ptr, ffn_down_w_node,
        //                                       ffn_down_node.find(TFF_GRAPH_NODE_CPU2GPU)->second);
        // tff::core::graph::NodeMetadata meta_ffn_down_b_node{node_name + "ffn_down_b_node"};
        // ffn_down_b_node->set_node_meta(meta_ffn_down_b_node);

        ffn_down_node[TFF_GRAPH_NODE_COMPUTE] = ffn_down_w_node;
    }

    //
    void LLAMACreator::build_ffn(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                     tff::core::graph::GraphNode> > &layer_map,
                                 std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                 NodeType &input_node,
                                 NodeType &ffn_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_NORM)->second;
        std::string node_name = layer->name();

        build_cpu_node(layer, graph_ptr, ffn_node);
        build_gpu_node(node_name, graph_ptr, ffn_node);

        auto rms_norm_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
        NodeMetadata meta_ffn_norm_node{node_name + "_ffn_rms_norm_node"};
        rms_norm_node->bind_devices(layer->device());
        rms_norm_node->set_node_meta(meta_ffn_norm_node);
        graph_ptr->add_node(rms_norm_node);
        graph_ptr->add_edge(input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second,
                            rms_norm_node);
        graph_ptr->add_edge(ffn_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU)->second,
                            rms_norm_node);


        // auto ffn_norm_w_node = build_mul_node(graph_ptr, ffn_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
        //                                       rms_norm_node);
        // tff::core::graph::NodeMetadata meta_ffn_norm_w_node{node_name + "_ffn_norm_mul_w_node"};
        // ffn_norm_w_node->set_node_meta(meta_ffn_norm_w_node);

        // auto ffn_norm_b_node = build_add_node(graph_ptr, ffn_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
        //                                  ffn_norm_w_node);
        // tff::core::graph::NodeMetadata meta_ffn_norm_b_node{node_name + "_ffn_norm_b_node"};
        // ffn_norm_b_node->set_node_meta(meta_ffn_norm_b_node);

        ffn_node[TFF_GRAPH_NODE_COMPUTE] = rms_norm_node;

        NodeType ffn_up_node;
        build_ffn_up(layer_map, graph_ptr, ffn_node, ffn_up_node);

        NodeType ffn_gate_node;
        build_ffn_gate(layer_map, graph_ptr, ffn_up_node, ffn_gate_node);

        NodeType ffn_down_node;
        build_ffn_down(layer_map, graph_ptr, ffn_gate_node, ffn_down_node);

        ffn_node[TFF_GRAPH_NODE_COMPUTE] = ffn_down_node.find(TFF_GRAPH_NODE_COMPUTE)->second;
    }

    //
    void LLAMACreator::build_output_norm(
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::graph::GraphNode> > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        NodeType &input_node,
        NodeType &output_norm_node) {
        auto layer = layer_map.find(tff::core::memory::ModelTensorType::LLM_TENSOR_OUTPUT_NORM)->second;
        std::string node_name = layer->name();

        //加载 norm层权重;
        build_cpu_node(layer, graph_ptr, output_norm_node);
        build_gpu_node(node_name, graph_ptr, output_norm_node);


        //
        auto rms_norm_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
        graph_ptr->add_node(rms_norm_node);
        rms_norm_node->bind_devices(layer->device());
        tff::core::graph::NodeMetadata meta_rms_norm_node{node_name + "_output_rms_norm_node"};
        rms_norm_node->set_node_meta(meta_rms_norm_node);
        graph_ptr->add_edge(input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second,
                            rms_norm_node);
        graph_ptr->add_edge(output_norm_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU)->second,
                            rms_norm_node);

        // auto output_norm_w_node = build_mul_node(graph_ptr,
        //                                          output_norm_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
        //                                          rms_norm_node);
        // tff::core::graph::NodeMetadata meta_output_norm_w_node{node_name + "_output_norm_w_node"};
        // output_norm_w_node->set_node_meta(meta_output_norm_w_node);

        // auto output_norm_b_node = build_add_node(graph_ptr,
        //                                          output_norm_node.find(TFF_GRAPH_NODE_CPU2GPU)->second,
        //                                          output_norm_w_node);
        // tff::core::graph::NodeMetadata meta_output_norm_b_node{node_name + "_output_norm_b_node"};
        // output_norm_b_node->set_node_meta(meta_output_norm_b_node);

        output_norm_node.insert({TFF_GRAPH_NODE_COMPUTE, rms_norm_node});
    }
}
