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
#include "model/layer/ModelLayer.h"
using namespace tff::core::global;
using namespace tff::core::graph;

namespace tff::core::model {
    class QWen3Creator final : public ModelCreatorBase {
    public:
        QWen3Creator() = default;

        ~QWen3Creator() override = default;
        //
        inline const char *get_model_name() override {
            return tff::core::model::to_string(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_QWEN3);
        }

        //
        inline void build_model_context(const model::GraphContext &ctx) override {
            this->_model_ctx = ctx;
        }
        void build_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) override;

        void build_mem_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) override;

        std::shared_ptr<tff::core::graph::GraphNode> build_inputs(
            const std::unordered_map<memory::ModelTensorType, std::shared_ptr<layer::ModelLayerObject>> &layer_map);

        std::shared_ptr<tff::core::graph::GraphNode> build_norm(memory::ModelTensorType tensor_type,
                                                                const std::unordered_map<memory::ModelTensorType, std::shared_ptr<layer::
                                                                ModelLayerObject>> &layer_map,
                                                                std::shared_ptr<GraphNode> &input_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_output(tff::core::memory::ModelTensorType tensor_type,
                                                                  const std::unordered_map<
                                                                      tff::core::memory::ModelTensorType,
                                                                      std::shared_ptr<
                                                                          tff::core::model::layer::ModelLayerObject> > &
                                                                  layer_map,
                                                                  std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                                                  std::shared_ptr<tff::core::graph::GraphNode> &
                                                                  input_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_reshape_node(memory::ModelTensorType tensor_type,
                                                                        const std::unordered_map<memory::ModelTensorType, std::shared_ptr<layer::
                                                                        ModelLayerObject>> &layer_map,
                                                                        std::shared_ptr<GraphNode> &input_node, int dim0, int dim1, int dim2);

        std::shared_ptr<tff::core::graph::GraphNode> build_mul_mat_node(
            std::shared_ptr<layer::ModelLayerObject> &layer,
            std::shared_ptr<GraphNode> &b_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_qkv_node(memory::ModelTensorType tensor_type,
                                                                    const std::unordered_map<memory::ModelTensorType, std::shared_ptr<layer::
                                                                    ModelLayerObject>>
                                                                    &layer_map,
                                                                    std::shared_ptr<GraphNode> &input_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_attn(
            const std::unordered_map<memory::ModelTensorType, std::shared_ptr<layer::ModelLayerObject>> &layer_map,
            std::shared_ptr<GraphNode> &q_node,
            std::shared_ptr<GraphNode> &k_node,
            std::shared_ptr<GraphNode> &v_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_add_node(
            std::shared_ptr<GraphNode> &a_node,
            std::shared_ptr<GraphNode> &b_node, bool inplace = false);

        std::shared_ptr<tff::core::graph::GraphNode> build_rope_node(
            std::shared_ptr<GraphNode> &input_node, std::shared_ptr<GraphNode> &rope_table_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_ffn_inp(
            std::shared_ptr<GraphNode> &input_node,
            std::shared_ptr<GraphNode> &current_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_ffn_up(
            const std::unordered_map<tff::core::memory::ModelTensorType,
                std::shared_ptr<tff::core::model::layer::ModelLayerObject> > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            std::shared_ptr<tff::core::graph::GraphNode> &input_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_ffn_gate(
            const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::model::layer::ModelLayerObject> > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            std::shared_ptr<tff::core::graph::GraphNode> &input_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_ffn_down(
            const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::model::layer::ModelLayerObject> > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            std::shared_ptr<tff::core::graph::GraphNode> &input_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_ffn(
            const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::model::layer::ModelLayerObject> > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
            std::shared_ptr<tff::core::graph::GraphNode> &input_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_unary_op(
            std::shared_ptr<tff::core::graph::GraphNode> &up_node,
            std::shared_ptr<tff::core::graph::GraphNode> &gate_node);

        std::shared_ptr<tff::core::graph::GraphNode> build_rope_table_node();

        std::shared_ptr<tff::core::graph::GraphNode> build_mul_node(
            std::shared_ptr<layer::ModelLayerObject> &layer, std::shared_ptr<GraphNode> &a_node);


        //build mem graph;
        //
        std::shared_ptr<tff::core::graph::GraphNode> build_host_node(
            std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node);

        //
        std::shared_ptr<tff::core::graph::GraphNode> build_device_node(
            std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node, std::shared_ptr<GraphNode> &current_cpu_node, bool
            is_input = false);
        //
        NodeType  build_layer_node(memory::ModelTensorType tensor_type,
            const std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
            tff::core::model::layer::ModelLayerObject> > &layer_map,NodeType &input_node, bool
            is_input = false);



    };
}
#endif //TFFINFER_LLAMACREATOR_H
