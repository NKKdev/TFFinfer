//
// Created by nkk on 2026/1/8.
//

#include "QWen3Creator.h"

namespace tff::core::model {
    void QWen3Creator::build_mem_graph(
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t, std::unordered_map<tff::
            core::memory::ModelTensorType, std::shared_ptr<tff::core::model::layer::ModelLayerObject> > > > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
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

        NodeType input_node;
        input_node = build_layer_node(memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD,
            input_layer_iter->second.begin()->second, input_node, true);

        if (input_layer_iter != layer_map.end() && !input_layer_iter->second.empty() &&
           repeating_layer_iter != layer_map.end() && !repeating_layer_iter->second.empty()) {
            //
            if (repeating_layer_iter != layer_map.end() && repeating_layer_iter->second.size() > 1) {
                const auto &repeating_layer_map = repeating_layer_iter->second;

                for (size_t layer_id = 0; layer_id < repeating_layer_map.size(); ++layer_id) {
                    auto &layer_map = repeating_layer_map.find(layer_id)->second;
                    //
                    auto attn_norm_node = build_layer_node(memory::ModelTensorType::LLM_TENSOR_ATTN_NORM,
                                                     layer_map, input_node);

                    auto attn_q_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q,
                                                          layer_map, attn_norm_node);
                    auto attn_k_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                                                      layer_map, attn_q_node);
                    auto attn_v_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V,
                                                      layer_map, attn_k_node);

                    auto attn_q_norm_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q_NORM,
                                                      layer_map, attn_v_node);
                    auto attn_k_norm_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K_NORM,
                                                     layer_map, attn_q_norm_node);

                    auto attn_ffn_norm_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_NORM,
                                                    layer_map, attn_k_norm_node);

                    auto attn_ffn_up_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_UP,
                                                    layer_map, attn_ffn_norm_node);

                    auto attn_ffn_gate_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_GATE,
                                                    layer_map, attn_ffn_up_node);

                    auto attn_ffn_down_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_DOWN,
                                                    layer_map, attn_ffn_gate_node);
                    input_node = attn_ffn_down_node;

                }
            }
        }
        auto output_norm_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_OUTPUT_NORM,
                                                   output_layer_iter->second.begin()->second, input_node);
        auto output_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_OUTPUT,
                                                   output_layer_iter->second.begin()->second, output_norm_node);

        graph_ptr->build_graph(output_node.find(GraphNodeType::TFF_GRAPH_NODE_CPU2GPU)->second);
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_host_node(
        std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node) {
        NodeType out_put_node;
        auto current_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
        tff::core::graph::NodeMetadata meta_map2cpu{layer->_layer_name + "_map2cpu"};
        current_map2cpu_node->set_node_meta(meta_map2cpu);
        auto device = tff::factory::ModuleFactory::instance()->create_shared<
            tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
        std::vector<int> device_ids;
        device->get_device_id(device_ids);
        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject>> devices = {{device_ids[0], device}};
        current_map2cpu_node->bind_devices(devices);
        auto params = current_map2cpu_node->get_params();
        params->set_param<size_t>(std::move(layer->_model_file_index));
        params->set_param<size_t>(std::move(layer->_offset));
        params->set_param<double>(std::move(layer->_data_size));
        params->set_param(this->_model_ctx._model_loader);
        current_map2cpu_node->add_src_node(input_node[tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU]);
        current_map2cpu_node->set_tensor(layer->_tensor);
        return current_map2cpu_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_device_node(
        std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node,
        std::shared_ptr<tff::core::graph::GraphNode> &current_cpu_node, bool is_input) {
        auto current_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
        current_cpu2gpu_node->set_node_meta(NodeMetadata{layer->_layer_name + "_cpu2gpu"});
        if (!is_input) {
            current_cpu2gpu_node->bind_devices(layer->_device_list);
        }else {
            auto device = tff::factory::ModuleFactory::instance()->create_shared<
            tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
            std::vector<int> device_ids;
            device->get_device_id(device_ids);
            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject>> devices = {{device_ids[0], device}};
            current_cpu2gpu_node->bind_devices(devices);
        }


        auto iter = input_node.find(GraphNodeType::TFF_GRAPH_NODE_CPU2GPU);
        if (iter != input_node.end()) {
            current_cpu2gpu_node->add_src_node(iter->second);
        }
        auto src_type = device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU;
        auto dst_type = device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU;
        auto params = current_cpu2gpu_node->get_params();
        params->set_param(make_cpy_kind(src_type, dst_type));

        current_cpu2gpu_node->add_src_node(current_cpu_node);
        auto tensor = std::make_shared<memory::Tensor>(layer->_tensor);
        current_cpu2gpu_node->set_tensor(tensor);
        return current_cpu2gpu_node;

    }

    NodeType QWen3Creator::build_layer_node(memory::ModelTensorType tensor_type,
        const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
        tff::core::model::layer::ModelLayerObject> > &layer_map, NodeType &input_node, bool
        is_input) {
        auto layer = layer_map.find(tensor_type)->second;
        if (layer == nullptr) {
            return NodeType();
        }
        NodeType out_put_node;
        out_put_node[TFF_GRAPH_NODE_MAP2CPU] = build_host_node(layer, input_node);
        out_put_node[TFF_GRAPH_NODE_CPU2GPU] = build_device_node(layer, input_node, out_put_node[TFF_GRAPH_NODE_MAP2CPU], is_input);

        return out_put_node;
    }
}
