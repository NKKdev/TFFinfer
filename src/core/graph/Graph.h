//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_DAGGRAPH_H
#define TFFINFER_DAGGRAPH_H

#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include "GraphNode.h"

namespace tff::core::graph {
    /**
     * @brief DAG Graph
     */
    class Graph final : public std::enable_shared_from_this<Graph> {
    public:
        Graph() = default;

        ~Graph() = default;

    public:
        /**
         * @brief Build graph
         * @param output_node
         */
        void build_graph(std::shared_ptr<GraphNode> output_node);

        /**
         * @brief 获取叶子节点
         * @return leafs
         */
        const std::vector<std::shared_ptr<GraphNode> > &leafs() const { return _leafs; }
        /**
         * @brief 获取所有节点
         * @return nodes
         */
        const std::vector<std::shared_ptr<GraphNode> > &nodes() const { return _nodes; }
        /**
         * @brief 获取所有节点
         * @return nodes
         */
        const std::vector<std::shared_ptr<GraphNode> > &total_nodes() const { return this->_total_nodes; }
        /**
         * @brief 获取节点使用次数
         * @return use_counts
         */
        const std::unordered_map<std::shared_ptr<GraphNode>, size_t> &use_counts() const { return _use_counts; }
        /**
         * @brief 清空
         */
        void clear();

        /**
         * @brief 检测是否有环
         * @param output_node
         * @return
         */
        bool has_cycle(std::shared_ptr<GraphNode> output_node);

        /**
         * @brief 分析生命周期
         */
        void analyze_lifetimes();

        /**
         * @brief 获取生命周期
         * @param node
         * @return
         */
        std::pair<int, int> get_lifetime(const std::shared_ptr<GraphNode> &node) const;

        /**
         * @brief 获取节点使用次数
         * @param node
         * @return
         */
        inline int get_use_count(std::shared_ptr<GraphNode> &node) {
            if (this->_use_counts.find(node) != this->_use_counts.end()) {
                return this->_use_counts[node];
            } else {
                return -1;
            }
        }

        /**
         * @brief 获取节点索引
         * @param node
         * @return
         */
        inline int get_node_index(const std::shared_ptr<GraphNode> &node) const {
            auto it = _node_to_index.find(node);
            return (it != _node_to_index.end()) ? static_cast<int>(it->second) : -1;
        }

        /**
         * @brief 获取叶子索引
         * @param node
         * @return
         */
        inline int get_leaf_index(const std::shared_ptr<GraphNode> &node) const {
            auto it = _leaf_to_index.find(node);
            return (it != _leaf_to_index.end()) ? static_cast<int>(it->second) : -1;
        }

        /**
         * @brief 获取输出节点
         * @return
         */
        inline std::vector<std::shared_ptr<GraphNode> > &output() {
            return this->_output_node;
        }

        /**
         * @brief 添加节点
         * @param node
         */
        inline void add_node(std::shared_ptr<graph::GraphNode> &node) {
            this->_nodes.push_back(node);
            _total_nodes.push_back(node);
        }

        /**
         * @brief 删除节点
         * @param node
         */
        inline void remove_node(const std::shared_ptr<GraphNode> &node) {
            auto iter = std::find(this->_nodes.begin(), this->_nodes.end(), node);
            this->_nodes.erase(iter);
        }

    private:
        /**
         * @brief 访问节点
         * @param node
         */
        void visit(std::shared_ptr<GraphNode> node);

        /**
         * @brief 是否是叶子节点
         * @param node
         * @return
         */
        bool is_leaf_node(const std::shared_ptr<GraphNode> &node) const;

    private:
        std::vector<std::shared_ptr<GraphNode> > _output_node;
        std::vector<std::shared_ptr<GraphNode> > _leafs;
        std::vector<std::shared_ptr<GraphNode> > _nodes;
        std::vector<std::shared_ptr<GraphNode> > _total_nodes;
        std::unordered_map<std::shared_ptr<GraphNode>, int> _exec_time;
        std::unordered_map<std::shared_ptr<GraphNode>, size_t> _use_counts;
        std::unordered_set<std::shared_ptr<GraphNode> > _visited;

        std::unordered_map<std::shared_ptr<GraphNode>, size_t> _node_to_index;
        std::unordered_map<std::shared_ptr<GraphNode>, size_t> _leaf_to_index;

    private:
        std::unordered_map<std::shared_ptr<GraphNode>, int> _first_use;
        std::unordered_map<std::shared_ptr<GraphNode>, int> _last_use;
    };
} // namespace tff::core::graph

#endif // TFFINFER_DAGGRAPH_H
