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
using namespace tff::core::graph;
namespace tff::core::model {
    class LLAMACreator : public ModelCreatorBase<LLAMACreator> {
#define ADD_NODE(enum_op_type) \
        tff::factory::ModuleFactory::instance()->create_shared<tff::core::graph::GraphNode>(OP_NODE_FLAG,\
        tff::factory::ModuleKeyType(enum_op_type));
        using NodeType = std::unordered_map<tff::core::graph::GraphNodeType,std::shared_ptr<graph::GraphNode>>;

    public:
        using callback_para = void(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &,
            std::shared_ptr<tff::core::graph::Graph> &);

    public:
        //
        static void build_mul_mat_node(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &_layer_map, std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                    std::shared_ptr<tff::core::graph::GraphNode> &input_node,
                    std::shared_ptr<tff::core::graph::GraphNode> &out_put_node) {
            out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MUL_MAT);
            graph_ptr->add_node(out_put_node);
            graph_ptr->add_edge(input_node, out_put_node);
        }
        //
        static void build_add_node(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &_layer_map, std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                    std::shared_ptr<tff::core::graph::GraphNode> &input_node,
                    std::shared_ptr<tff::core::graph::GraphNode> &out_put_node) {
            out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ADD);
            graph_ptr->add_node(out_put_node);
            graph_ptr->add_edge(input_node, out_put_node);
        }
        //
        static void build_cpu_node(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &_layer_map, std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                    NodeType &input_node,
                    NodeType &out_put_node) {
            auto current_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
            graph_ptr->add_node(current_map2cpu_node);
            graph_ptr->add_edge(input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU)->second,
                                current_map2cpu_node);
            out_put_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU, current_map2cpu_node});
        }
        //
        static void build_gpu_node(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &_layer_map, std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                    NodeType &input_node,
                    NodeType &out_put_node) {
            auto current_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
            graph_ptr->add_node(current_cpu2gpu_node);
            graph_ptr->add_edge(out_put_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU)->second,
                                current_cpu2gpu_node);

            auto current_compute_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_REF);
            graph_ptr->add_node(current_compute_node);
            graph_ptr->add_edge(current_cpu2gpu_node, current_compute_node);
            out_put_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU, current_cpu2gpu_node});
            out_put_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE, current_compute_node});
        }
        //
        static void build_attn_norm(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &layer_map, std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                    NodeType &input_node,
                    NodeType &attn_norm_node) {

            build_cpu_node(layer_map, graph_ptr, input_node, attn_norm_node);
            build_gpu_node(layer_map, graph_ptr, attn_norm_node, attn_norm_node);
            //
            auto rms_norm_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
            graph_ptr->add_node(rms_norm_node);
            graph_ptr->add_edge(input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second,
                                rms_norm_node);

            build_mul_mat_node(layer_map, graph_ptr, rms_norm_node,
                attn_norm_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
        }
        //
        static void build_inputs(
            const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &_layer_map, std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                    NodeType &input_node) {
            //
            auto token_embd_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
            token_embd_map2cpu_node->set_name("token_embding_map2cpu_node");
            graph_ptr->add_node(token_embd_map2cpu_node);
            auto token_embd_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
            token_embd_cpu2gpu_node->set_name("token_embding_cpu2gpu_node");
            graph_ptr->add_node(token_embd_cpu2gpu_node);
            auto tokenize_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_TOKENIZE);
            tokenize_node->set_name("tokenize_node");
            graph_ptr->add_node(tokenize_node);
            graph_ptr->add_edge(token_embd_map2cpu_node, token_embd_cpu2gpu_node);
            graph_ptr->add_edge(token_embd_cpu2gpu_node, tokenize_node);

            std::unordered_map<tff::core::graph::GraphNodeType,std::shared_ptr<graph::GraphNode>> input;

            //
            input_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE, tokenize_node});
            input_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU, token_embd_map2cpu_node});
            input_node.insert({tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU, token_embd_cpu2gpu_node});
        }
        //
        static void build_qkv_node(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::graph::GraphNode> > > > &layer_map, std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                    NodeType &input_node,
                    NodeType &attn_qkv_node) {
            build_cpu_node(layer_map, graph_ptr, input_node, attn_qkv_node);
            build_gpu_node(layer_map, graph_ptr, attn_qkv_node, attn_qkv_node);
            //
            build_mul_mat_node(layer_map, graph_ptr,
                input_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second,
                attn_qkv_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second);

            auto q_reshape_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
            graph_ptr->add_node(q_reshape_node);
            graph_ptr->add_edge(attn_qkv_node.find(tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_COMPUTE)->second,
                                q_reshape_node);
        }
        //
        static void build_attn(const std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                                    std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                        tff::core::graph::GraphNode> > > > &layer_map,
                                        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                        NodeType &input_node,
                                        NodeType& q_node,
                                        NodeType& k_node,
                                        NodeType& v_node,
                                        NodeType& out_put_node){
            auto flash_attn_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT);
            graph_ptr->add_node(flash_attn_node);
            graph_ptr->add_edge(input_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
            graph_ptr->add_edge(q_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
            graph_ptr->add_edge(k_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
            graph_ptr->add_edge(v_node.find(TFF_GRAPH_NODE_COMPUTE)->second, flash_attn_node);
            out_put_node.insert({TFF_GRAPH_NODE_COMPUTE, flash_attn_node});
        }
        //
        static void add_node();
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
                                        tff::core::graph::GraphNode> > > > &layer_map,
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

            const auto input_layer_iter = layer_map.find(
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT);
            const auto repeating_layer_iter = layer_map.find(
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
            //
            const auto output_layer_iter = layer_map.find(
                tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT);

            NodeType input_node;
            build_inputs(layer_map, graph_ptr,input_node);

            //
            if (input_layer_iter != layer_map.end() && !input_layer_iter->second.empty() &&
                repeating_layer_iter != layer_map.end() && !repeating_layer_iter->second.empty()) {
                //auto &input_layers = input_layer_iter->second;
                //auto &input_node = input_layers.begin()->second.find(tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD)->second;
                //
                if (repeating_layer_iter != layer_map.end() && repeating_layer_iter->second.size() > 1) {
                    const auto &repeating_layer_map = repeating_layer_iter->second; // Assume it's a std::set or similar
                    for (size_t layer_id = 0; layer_id < repeating_layer_map.size(); ++layer_id) {
                        tff::log::Logger::info("build layer :%d graph\n", layer_id);
                        auto &layer = repeating_layer_map.find(layer_id)->second;
                        //
                        NodeType attn_norm_node;
                        build_attn_norm(layer_map, graph_ptr, input_node, attn_norm_node);
                        //process qkv weight
                        {
                            NodeType attn_q_node;
                            build_qkv_node(layer_map, graph_ptr, attn_norm_node, attn_q_node);
                            NodeType attn_k_node;
                            build_qkv_node(layer_map, graph_ptr, attn_norm_node, attn_k_node);
                            NodeType attn_v_node;
                            build_qkv_node(layer_map, graph_ptr, attn_norm_node, attn_v_node);

                            //
                            auto q_rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
                            graph_ptr->add_node(q_rope_node);
                            graph_ptr->add_edge(attn_q_node.find(TFF_GRAPH_NODE_COMPUTE)->second, q_rope_node);
                            auto q_compute_node = ADD_NODE(TFF_OP_MEM_REF);
                            graph_ptr->add_node(q_compute_node);
                            graph_ptr->add_edge(q_rope_node, q_compute_node);
                            attn_q_node.insert({TFF_GRAPH_NODE_COMPUTE, q_compute_node});
                            //
                            auto k_rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
                            add_nodes_to_graph(k_rope_node);
                            graph_ptr->add_edge(attn_k_node.find(TFF_GRAPH_NODE_COMPUTE)->second, k_rope_node);
                            auto k_compute_node = ADD_NODE(TFF_OP_MEM_REF);
                            graph_ptr->add_node(k_compute_node);
                            graph_ptr->add_edge(k_rope_node, k_compute_node);
                            attn_k_node.insert({TFF_GRAPH_NODE_COMPUTE, k_compute_node});

                            //
                            NodeType attn_node;
                            build_attn(layer_map, graph_ptr, input_node, attn_q_node, attn_k_node, attn_v_node, attn_node);
                            //
                            build_add_node(layer_map, graph_ptr, attn_node.find(TFF_GRAPH_NODE_COMPUTE)->second,
                                input_node.find(TFF_GRAPH_NODE_COMPUTE)->second);
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
