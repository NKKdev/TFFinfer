//
// Created by nkk on 2026/1/8.
//

#include "QWen3Creator.h"
#include "include/TFFOPCreator.h"

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

                for (int layer_id = 0; layer_id < repeating_layer_map.size(); ++layer_id) {
#ifdef _DEBUG1
                    {
                        if (layer_id >= 1) {
                            continue;
                        }
                    }
#endif
                    input_node[TFF_GRAPH_NODE_CPU2GPU]->set_layer_id(layer_id);
                    input_node[TFF_GRAPH_NODE_MAP2CPU]->set_layer_id(layer_id);
                    auto &repeate_layer_map = repeating_layer_map.find(layer_id)->second;
                    //
                    auto attn_norm_node = build_layer_node(memory::ModelTensorType::LLM_TENSOR_ATTN_NORM,
                                                           repeate_layer_map, input_node);

                    auto attn_q_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q,
                                                        repeate_layer_map, attn_norm_node);
                    auto attn_k_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                                                        repeate_layer_map, attn_q_node);
                    auto attn_v_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V,
                                                        repeate_layer_map, attn_k_node);

                    auto attn_q_norm_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q_NORM,
                                                             repeate_layer_map, attn_v_node);
                    auto attn_k_norm_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K_NORM,
                                                             repeate_layer_map, attn_q_norm_node);

                    auto attn_output_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_OUT,
                                                             repeate_layer_map, attn_k_norm_node);

                    auto attn_ffn_norm_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_NORM,
                                                               repeate_layer_map, attn_output_node);

                    auto attn_ffn_up_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_UP,
                                                             repeate_layer_map, attn_ffn_norm_node);

                    auto attn_ffn_gate_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_GATE,
                                                               repeate_layer_map, attn_ffn_up_node);

                    auto attn_ffn_down_node = build_layer_node(tff::core::memory::ModelTensorType::LLM_TENSOR_FFN_DOWN,
                                                               repeate_layer_map, attn_ffn_gate_node);
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
}
