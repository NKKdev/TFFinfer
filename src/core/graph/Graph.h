//
// Created by nkk on 2025/10/24.
//

#ifndef TFFINFER_DAGGRAPH_H
#define TFFINFER_DAGGRAPH_H
#include <memory>
#include <queue>
#include <set>
#include <unordered_set>
#include "GraphNode.h"
namespace tff::core::graph {

    class Graph final : public std::enable_shared_from_this<Graph> {
    public:
        Graph() = default;

        virtual ~Graph() {
            clear();
        }

    public:
        //
        inline bool add_node(std::shared_ptr<GraphNode> node) {
            if (!node) return false;
            std::lock_guard<std::mutex> lock(_mutex);
            if (_node_set.find(node) != _node_set.end()) {
                return false; // 已存在
            }
            _nodes.push_back(node);
            _node_set.insert(node);
            if (node->is_input_node()) {
                _input_node = (node);
            }
            if (node->is_output_node()) {
                _output_node = (node);
            }
            return true;
        }

        // 建立边：from -> to
        bool add_edge(std::shared_ptr<GraphNode> &from, std::shared_ptr<GraphNode> &to);

        // 构建拓扑排序（Kahn 算法）
        std::vector<std::shared_ptr<GraphNode> > topological_sort();

        // 前向执行整个图
        bool forward();
        // 释放所有资源
        inline void release() const {
            std::lock_guard<std::mutex> lock(_mutex);
            for (const auto &node: _nodes) {
                node->release();
            }
        }

        // 清空图
        inline void clear() {
            std::lock_guard<std::mutex> lock(_mutex);
            release();
            _nodes.clear();
            _node_set.clear();
        }

        // 获取所有节点
        inline const std::vector<std::shared_ptr<GraphNode> > &nodes() const {
            return _nodes;
        }

        // 调试：打印拓扑结构
        void print_topology() const;

        // 检查是否有环（用于调试）
        inline bool has_cycle() {
            return topological_sort().empty() && !_nodes.empty();
        }
        //
        inline std::shared_ptr<tff::core::graph::GraphNode> get_input_nodes() {
            std::lock_guard<std::mutex> lock(_mutex);
            return _input_node;
        }

    private:
        inline bool has_edge(const std::shared_ptr<GraphNode> &from, const std::shared_ptr<GraphNode> &to) const {
            for (const auto &next_weak: from->_next_nodes) {
                if (auto next = next_weak.lock()) {
                    if (next.get() == to.get()) {
                        return true;
                    }
                }
            }
            return false;
        }

    private:
        std::vector<std::shared_ptr<GraphNode> > _nodes;
        std::unordered_set<std::shared_ptr<GraphNode> > _node_set; // 去重
        //
        std::shared_ptr<GraphNode> _input_node;
        std::shared_ptr<GraphNode> _output_node;
        mutable std::mutex _mutex;
        //

    };
}


#endif //TFFINFER_DAGGRAPH_H
