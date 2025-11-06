//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_LLAMACREATOR_H
#define TFFINFER_LLAMACREATOR_H
#include "model/base/ModelCreatorBase.h"
#include <memory>

#include "mem/MemBufferAllocatorBaseObject.h"
#include "graph/GraphNode.h"
#include "global/ModelGlobalVar.h"
#include "global/GlobalDefine.h"
#include "FunctionFactory.h"
using namespace tff::core::global;

namespace tff::core::model {
    class LLAMACreator : public ModelCreatorBase<LLAMACreator> {
#define ADD_NODE(enum_op_type) \
        tff::factory::ModuleFactory::instance()->create_shared<tff::core::graph::GraphNode>(OP_NODE_FLAG,\
        tff::factory::ModuleKeyType(enum_op_type));

    public:
        using callback_para = void(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &,
            std::shared_ptr<tff::core::graph::Graph> &);

    public:
        static void build_attn(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                                    std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                        tff::core::graph::GraphNode> > > > &_layer_map,
                                        const std::shared_ptr<tff::core::graph::GraphNode>& q_node,
            const std::shared_ptr<tff::core::graph::GraphNode>& k_node,
            const std::shared_ptr<tff::core::graph::GraphNode>& v_node) {

        }
    public:
        static void create_layer(std::shared_ptr<tff::core::memory::Tensor> &tensor_ptr,
                                 std::shared_ptr<tff::core::graph::GraphNode> &layer_node,
                                 const size_t &total_layer_num = -1, const size_t &layer_index = -1) {
            auto &layer_info = LLM_LAYER_OP_INFOS.find(tensor_ptr->get_tensor_type())->second;
            layer_node = tff::factory::ModuleFactory::instance()->create_shared<tff::core::graph::GraphNode>(
                OP_NODE_FLAG, layer_info.second);
            if (!layer_node) {
                return;
            }
            layer_node->set_layer_id(layer_index);
            layer_node->set_layer_type(layer_info.first);
            std::vector<std::shared_ptr<tff::core::memory::Tensor> > _src_tensors_ptr;
            _src_tensors_ptr.push_back(tensor_ptr);
            layer_node->set_inputs(_src_tensors_ptr);
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
                    auto device_size = global::get_device_size(DEVICE_BACKEND_TYPE_CUDA);
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
                    const int layer_gpu = std::upper_bound(device_splits.begin(), device_splits.begin() + device_size,
                                                           float(layer_index) / total_layer_num) - device_splits.
                                          begin();
                    layer_node->bind_devices(device_cuda); //todo 应该绑定某种类型设备下某个设备
                    break;
                }
            }
        }

        //
        static void build_graph(std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                                    std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                        tff::core::graph::GraphNode> > > > &_layer_map,
                                std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
            //
            if (!graph_ptr) {
                graph_ptr = std::make_shared<tff::core::graph::Graph>();
            }
            //
            auto add_nodes_to_graph = [&graph_ptr](
                const std::shared_ptr<tff::core::graph::GraphNode> &node) {
                if (node) {
                    graph_ptr->add_node(node);
                }
            };
            //
            for (auto &layers: _layer_map) {
                for (auto &one_layer: layers.second) {
                    for (auto &one_node: one_layer.second) {
                        add_nodes_to_graph(one_node.second);
                    }
                }
            }
            const auto input_layer_iter = _layer_map.find(
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT);
            const auto repeating_layer_iter = _layer_map.find(
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
            //
            const auto output_layer_iter = _layer_map.find(
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT);

            //
            auto map2CPU_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
            add_nodes_to_graph(map2CPU_node);
            auto tokenize_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_TOKENIZE);
            add_nodes_to_graph(tokenize_node);
            auto token_embdCPU2GPU_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
            add_nodes_to_graph(token_embdCPU2GPU_node);
            graph_ptr->add_edge(map2CPU_node, token_embdCPU2GPU_node);

            //
            if (input_layer_iter != _layer_map.end() && !input_layer_iter->second.empty() &&
                repeating_layer_iter != _layer_map.end() && !repeating_layer_iter->second.empty()) {
                //auto &input_layers = input_layer_iter->second;
                //auto &input_node = input_layers.begin()->second.find(tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD)->second;
                //
                if (repeating_layer_iter != _layer_map.end() && repeating_layer_iter->second.size() > 1) {
                    const auto &repeating_layer_map = repeating_layer_iter->second; // Assume it's a std::set or similar
                    for (size_t layer_id = 0; layer_id < repeating_layer_map.size(); ++layer_id) {
                        tff::log::Logger::info("build layer :%d graph\n", layer_id);
                        auto &layer = repeating_layer_map.find(layer_id)->second;
                        //
                        auto attn_norm_weight_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
                        add_nodes_to_graph(attn_norm_weight_map2cpu_node);
                        graph_ptr->add_edge(map2CPU_node, attn_norm_weight_map2cpu_node);
                        //
                        auto attn_norm_weight_cpu2gpu = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
                        add_nodes_to_graph(attn_norm_weight_cpu2gpu);
                        graph_ptr->add_edge(attn_norm_weight_map2cpu_node, attn_norm_weight_cpu2gpu);
                        //
                        auto &attn_norm_node = layer.find(
                            tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_NORM)->second;
                        graph_ptr->add_edge(token_embdCPU2GPU_node, attn_norm_node);
                        // auto attn_norm_mul_attn_norm_w = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MUL_MAT);
                        // add_nodes_to_graph(attn_norm_mul_attn_norm_w);
                        // graph_ptr->add_edge(attn_norm_weight_cpu2gpu, attn_norm_mul_attn_norm_w);
                        // graph_ptr->add_edge(attn_norm_node, attn_norm_mul_attn_norm_w);
                        // auto add_attn_norm_bias_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ADD);
                        // add_nodes_to_graph(add_attn_norm_bias_node);
                        // graph_ptr->add_edge(attn_norm_mul_attn_norm_w, add_attn_norm_bias_node);
                        //process qkv weight
                        {
                            // auto q_w_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
                            // add_nodes_to_graph(q_w_map2cpu_node);
                            // graph_ptr->add_edge(attn_norm_weight_map2cpu_node, q_w_map2cpu_node);
                            // //
                            // auto q_w_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
                            // add_nodes_to_graph(q_w_cpu2gpu_node);
                            // graph_ptr->add_edge(q_w_map2cpu_node, q_w_cpu2gpu_node);
                            // auto &attn_q_node = layer.find(
                            //     tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q)->second;
                            // graph_ptr->add_edge(q_w_cpu2gpu_node, attn_q_node);
                            // graph_ptr->add_edge(add_attn_norm_bias_node, attn_q_node);
                            //
                            // //
                            // auto k_w_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
                            // add_nodes_to_graph(k_w_map2cpu_node);
                            // graph_ptr->add_edge(attn_norm_weight_map2cpu_node, k_w_map2cpu_node);
                            // //
                            // auto k_w_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
                            // add_nodes_to_graph(k_w_cpu2gpu_node);
                            // graph_ptr->add_edge(k_w_map2cpu_node, k_w_cpu2gpu_node);
                            //
                            // auto &attn_k_node = layer.find(
                            //     tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K)->second;
                            // graph_ptr->add_edge(k_w_cpu2gpu_node, attn_k_node);
                            // graph_ptr->add_edge(add_attn_norm_bias_node, attn_k_node);
                            //
                            //
                            // //
                            // auto v_w_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
                            // add_nodes_to_graph(v_w_map2cpu_node);
                            // graph_ptr->add_edge(attn_norm_weight_map2cpu_node, v_w_map2cpu_node);
                            // //
                            // auto v_w_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
                            // add_nodes_to_graph(v_w_cpu2gpu_node);
                            // graph_ptr->add_edge(v_w_map2cpu_node, v_w_cpu2gpu_node);
                            // auto &attn_v_node = layer.find(
                            //     tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V)->second;
                            // graph_ptr->add_edge(q_w_cpu2gpu_node, attn_v_node);
                            // graph_ptr->add_edge(add_attn_norm_bias_node, attn_v_node);
                            //
                            //
                            // //reshape;
                            // auto q_reshaper_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
                            // add_nodes_to_graph(q_reshaper_node);
                            // q_reshaper_node->set_layer_type(
                            //     tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
                            // q_reshaper_node->set_op_type(graph::TFF_OP_RESHAPE);
                            // q_reshaper_node->set_name("Q_reshaper");
                            // q_reshaper_node->set_layer_id(attn_q_node->layer_id());
                            // graph_ptr->add_edge(attn_q_node, q_reshaper_node);
                            // //
                            // auto k_reshaper_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
                            // add_nodes_to_graph(k_reshaper_node);
                            // k_reshaper_node->set_layer_type(
                            //     tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
                            // k_reshaper_node->set_op_type(graph::TFF_OP_RESHAPE);
                            // k_reshaper_node->set_name("Q_reshaper");
                            // k_reshaper_node->set_layer_id(attn_k_node->layer_id());
                            // graph_ptr->add_edge(attn_k_node, k_reshaper_node);
                            // //
                            // auto v_reshaper_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
                            // add_nodes_to_graph(v_reshaper_node);
                            // v_reshaper_node->set_layer_type(
                            //     tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
                            // v_reshaper_node->set_op_type(graph::TFF_OP_RESHAPE);
                            // v_reshaper_node->set_name("Q_reshaper");
                            // v_reshaper_node->set_layer_id(attn_v_node->layer_id());
                            // graph_ptr->add_edge(attn_v_node, v_reshaper_node);
                            // //
                            // auto q_rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
                            // add_nodes_to_graph(q_rope_node);
                            // graph_ptr->add_edge(q_reshaper_node, q_rope_node);
                            // //
                            // auto k_rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
                            // add_nodes_to_graph(k_rope_node);
                            // graph_ptr->add_edge(k_reshaper_node, k_rope_node);
                            //
                            // //
                            // build_attn(_layer_map, q_rope_node, k_rope_node, v_reshaper_node);
                            //
                        }
                    }
                }
            }
        }

        //
        static const char *get_model_name() {
            return tff::core::global::LLM_ARCH_NAMES.find(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA)
                    ->second;
        }
    };
}
#endif //TFFINFER_LLAMACREATOR_H
