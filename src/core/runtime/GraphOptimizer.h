//
// Created by nkk on 2026/2/18.
//

#ifndef TFFINFER_GRAPHOPTIMIZER_H
#define TFFINFER_GRAPHOPTIMIZER_H
#include <memory>

#include "ModuleObject.h"
#include "model/base/ModelCreatorBase.h"

namespace tff::core::graph {
    class GraphNode;
    class Graph;
}
namespace tff::core::runtime{
    /**
     * @brief 图优化器
     */
    class GraphOptimizer : public tff::module::ModuleObject{
    public:
        GraphOptimizer() = default;
        ~GraphOptimizer() = default;
    public:
        /**
         * @brief 图优化
         * @param graph_ctx 图上下文
         * @param graph_ptr  计算图
         */
        void optimize(GraphContext &_model_ctx, std::shared_ptr<graph::Graph> &graph_ptr);
    protected:
        /**
         * @brief 设备分配优化
         * @param graph_ctx 图上下文
         * @param graph_ptr  计算图
         */
        void device_placement_opt(GraphContext &graph_ctx, std::shared_ptr<graph::Graph> &graph_ptr);
        /**
         * @brief 消除死代码
         * @param graph_ptr  计算图
         */
        void elimination_dead_code(std::shared_ptr<core::graph::Graph> &graph_ptr);
        /**
         * @brief 优化内存，这里指层间显存复用
         * @param graph_ptr  计算图
         */
        void mem_opt(std::shared_ptr<core::graph::Graph> &graph_ptr);
        /**
         * @brief 融合节点
         * @param graph_ptr  计算图
         */
        void fuse(std::shared_ptr<core::graph::Graph> &graph_ptr) ;
        /**
         * @brief 检测是否可以融合
         * @param current_node 当前节点
         * @param pre_node 前一个节点
         * @return 是否可以融合
         */
        bool can_fuse(
            const std::shared_ptr<graph::GraphNode> &current_node,
            std::shared_ptr<graph::GraphNode> &pre_node) const;
        /**
         * @brief 融合节点
         * @param graph_ptr  计算图
         * @param current_node 当前节点
         * @return 是否融合成功
         */
        bool fuse(
            std::shared_ptr<graph::Graph> &graph_ptr, const std::shared_ptr<graph::GraphNode> &current_node);
        /**
         * @brief 量化节点融合(已废弃)
         * @param current_node 当前节点
         * @return 是否可以融合
         */
        bool fuse_quant_node(
            const std::shared_ptr<graph::GraphNode> &current_node) const;
        /**
         * @brief 融合相同节点
         * @param current_node 当前节点
         * @return 是否可以融合
         */
        bool fuse_same_node(const std::shared_ptr<graph::GraphNode> &current_node) const;
        /**
         * @brief 是否需要融合相同节点
         * @param current_node 当前节点
         * @return 是否需要融合相同节点
         */
        bool is_need_fuse_same_node(const std::shared_ptr<graph::GraphNode> &current_node) const;
        /**
         * @brief 融合相同节点
         * @param current_node 当前节点
         * @param remain_node 剩余节点
         * @param same_node 相同节点
         */
        void fuse_same_node(const std::shared_ptr<graph::GraphNode> &current_node,
                            const std::shared_ptr<graph::GraphNode> &remain_node,
                            const std::shared_ptr<graph::GraphNode> &same_node) const;
        /**
         * @brief 插入输入节点
         * @param current_node 当前节点
         * @param new_input_node 新输入节点
         * @param old_input_node 旧输入节点
         */
        void insert_input_node(const std::shared_ptr<graph::GraphNode> &current_node,
                               const std::shared_ptr<graph::GraphNode> &new_input_node,
                               const std::shared_ptr<graph::GraphNode> &old_input_node) const;
        /**
         * @brief 替换输入节点
         * @param current_node 当前节点
         * @param new_input_node 新输入节点
         * @param old_input_node 旧输入节点
         */
        void replace_input_node(const std::shared_ptr<graph::GraphNode> &current_node,
                            const std::shared_ptr<graph::GraphNode> &new_input_node,
                            const std::shared_ptr<graph::GraphNode> &old_input_node) const;
    };
}



#endif //TFFINFER_GRAPHOPTIMIZER_H