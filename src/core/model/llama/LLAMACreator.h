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
#include "ModuleFactory.h"
#include "graph/Graph.h"
#include "model/base/ModelLoaderBase.h"
using namespace tff::core::global;
using namespace tff::core::graph;

namespace tff::core::model {
    class LLAMACreator final : public ModelCreatorBase {
    public:
        LLAMACreator() = default;

        ~LLAMACreator() override = default;

    protected:
        //
        std::shared_ptr<tff::core::graph::GraphNode> build_rope_table_node(
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr);

        //
        std::shared_ptr<tff::core::graph::GraphNode> build_mul_node(
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            std::shared_ptr<tff::core::graph::GraphNode> &a_node, std::shared_ptr<tff::core::graph::GraphNode> &b_node);

        //
        std::shared_ptr<tff::core::graph::GraphNode> build_mul_mat_node(
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            std::shared_ptr<tff::core::graph::GraphNode> &a_node, std::shared_ptr<tff::core::graph::GraphNode> &b_node);

        //
        std::shared_ptr<tff::core::graph::GraphNode> build_add_node(
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            std::shared_ptr<tff::core::graph::GraphNode> &a_node,
            std::shared_ptr<tff::core::graph::GraphNode> &b_node);

        //
        void build_cpu_node(
            std::shared_ptr<GraphNode> &layer,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            NodeType &out_put_node);

        //
        void build_gpu_node(
            const std::string &node_name,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr, NodeType &out_put_node, bool is_input = false);

        //
        void build_attn_norm(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                 tff::core::graph::GraphNode> > &layer_map,
                             std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                             NodeType &input_node,
                             NodeType &attn_norm_node);

        //
        void build_inputs(
            const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::graph::GraphNode> > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            NodeType &input_node);

        //
        void build_qkv_node(std::shared_ptr<GraphNode> &layer,
                            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                            NodeType &input_node,
                            NodeType &attn_qkv_node);

        //
        void build_attn(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                            tff::core::graph::GraphNode> > &layer_map,
                        std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                        NodeType &input_node,
                        NodeType &q_node,
                        NodeType &k_node,
                        NodeType &v_node,
                        NodeType &out_put_node);

        void build_ffn_up(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                              tff::core::graph::GraphNode> > &layer_map,
                          std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                          NodeType &input_node,
                          NodeType &ffn_up_node);

        //
        void build_ffn_gate(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                tff::core::graph::GraphNode> > &layer_map,
                            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                            NodeType &input_node,
                            NodeType &ffn_gate_node);

        //
        void build_ffn_down(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                tff::core::graph::GraphNode> > &layer_map,
                            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                            NodeType &input_node,
                            NodeType &ffn_down_node);

        //
        void build_ffn(const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                           tff::core::graph::GraphNode> > &layer_map,
                       std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                       NodeType &input_node,
                       NodeType &ffn_node);

        //
        void build_output_norm(
            const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::graph::GraphNode> > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            NodeType &input_node,
            NodeType &output_norm_node);

        //
        inline void update_mem_node(const tff::core::graph::GraphNodeType &node_type,
                                    const std::shared_ptr<graph::GraphNode> &mem_node) {
            this->_current_mem_node[node_type] = mem_node;
        }

    public:
        void build_layer(std::shared_ptr<tff::core::memory::Tensor> &tensor_ptr,
                         std::shared_ptr<tff::core::graph::GraphNode> &layer_node,
                         const size_t &total_layer_num = -1, const size_t &layer_index = -1) override;

        //
        void build_graph(std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                             std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                 tff::core::graph::GraphNode> > > > &layer_map,
                         std::shared_ptr<tff::core::graph::Graph> &graph_ptr) override;

        //
        inline const char *get_model_name() override {
            return tff::core::global::LLM_ARCH_NAMES.find(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_LLAMA)
                    ->second;
        }

        //
        inline void set_loader(const std::shared_ptr<tff::core::model::ModelLoaderBase> &loader) override {
            this->_model_loader = loader;
        }

    protected:
        std::shared_ptr<tff::core::model::ModelLoaderBase> _model_loader;

    public:
        NodeType _current_mem_node;
        bool _is_input_norm_w = true;
        bool _is_input_norm_b = false;
        bool _is_ffn_norm_w = true;
        bool _is_ffn_norm_b = false;
        bool _is_attn_norm_w = true;
        bool _is_attn_norm_b = true;
        bool _is_output_norm_w = true;
        bool _is_output_norm_b = true;
    };
}
#endif //TFFINFER_LLAMACREATOR_H
