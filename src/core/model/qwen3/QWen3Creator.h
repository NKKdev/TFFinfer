//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_LLAMACREATOR_H
#define TFFINFER_LLAMACREATOR_H
#include "model/base/ModelCreatorBase.h"
#include <memory>

#include "../../device/MemBufferAllocatorBaseObject.h"
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
    /**
     * Qwen3Creator
     */
    class QWen3Creator final : public ModelCreatorBase {
    public:
        QWen3Creator() = default;

        ~QWen3Creator() override = default;

    public:
        /**
         * 获取模型名称
         * @return
         */
        inline const char *get_model_name() override {
            return tff::core::model::to_string(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_QWEN3);
        }

        /**
         * 构建模型上下文
         * @param ctx
         */
        inline void build_model_context(const model::GraphContext &ctx) override {
            this->_graph_ctx = ctx;
        }

    public:
        /**
         * 构建模型计算图
         * @param layer_map 模型层信息
         * @param graph_ptr 计算图对象
         */
        void build_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) override;

        /**
         * 构建模型内存IO计算图
         * @param layer_map 模型层信息
         * @param graph_ptr 模型内存图对象
         */
        void build_mem_graph(
            std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                    tff::core::model::layer::ModelLayerObject> > > > &layer_map,
            std::shared_ptr<tff::core::graph::Graph> &graph_ptr) override;

    protected:
        /**
         * 构建Rope表节点
         * @return
         */
        std::shared_ptr<GraphNode> build_rope_table_node();

        /**
         * @brief 构建输入节点
         * @param layer_map
         * @return
         */
        std::shared_ptr<GraphNode> build_inputs(
            const std::unordered_map<memory::ModelTensorType, std::shared_ptr<layer::ModelLayerObject> > &layer_map);

        /**
         * 构建输出节点
         * @param tensor_type 模型张量类型
         * @param input_node 输入节点
         * @return
         */
        std::shared_ptr<GraphNode> build_output(memory::ModelTensorType tensor_type,
                                                std::shared_ptr<GraphNode> &input_node);

        /**
         * @brief 构建QKV节点
         * @param weight_node 权重节点
         * @param x_node 输入节点
         * @return
         */
        std::shared_ptr<GraphNode> build_qkv_node(
            std::shared_ptr<GraphNode> &weight_node, std::shared_ptr<GraphNode> &x_node);

        /**
         * @brief 构建FFN输入节点
         * @param input_node 输入节点
         * @param current_node 当前节点
         * @return
         */
        std::shared_ptr<GraphNode> build_ffn_inp(
            std::shared_ptr<GraphNode> &input_node,
            std::shared_ptr<GraphNode> &current_node);

        /**
         * @brief 构建提取输出logits节点
         * @param node 输入节点
         * @return
         */
        std::shared_ptr<GraphNode> build_gather_node(std::shared_ptr<GraphNode> &node);

        /**
         * @brief 构建数据卸载节点， GPU卸载结果到CPU
         * @param node 输入节点
         * @return
         */
        std::shared_ptr<GraphNode> build_offload_node(std::shared_ptr<GraphNode> &node);

        /**
         * @brief 构建Rope节点
         * @param layer_id 层ID
         * @param input_node 输入节点
         * @return
         */
        std::shared_ptr<graph::GraphNode> build_rope_node(
            int layer_id, const std::shared_ptr<GraphNode> &input_node);
    };
}
#endif //TFFINFER_LLAMACREATOR_H
