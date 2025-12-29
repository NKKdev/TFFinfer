//
// Created by nkk on 2025/10/24.
//

#include "Graph.h"
#include "global/ModelGlobalVar.h"

namespace tff::core::graph {
    bool Graph::add_edge(std::shared_ptr<GraphNode> &from, std::shared_ptr<GraphNode> &to) {
        if (!from || !to) {
            tff::log::Logger::error("add_edge failed!! from or to is invalid!!");
            return false;
        }
        std::lock_guard<std::mutex> lock(_mutex);

        // 检查是否已存在
        if (has_edge(from, to)) {
            return true;
        }

        from->add_successors(to);
        to->add_predecessors(from);

        return true;
    }

    // 构建拓扑排序（Kahn 算法）
    std::vector<std::shared_ptr<GraphNode> > Graph::topological_sort() {
        std::lock_guard<std::mutex> lock(_mutex);
        std::vector<std::shared_ptr<GraphNode> > sorted;
        std::queue<std::shared_ptr<GraphNode> > q;
        std::unordered_map<GraphNode *, int> in_degree;

        // 计算入度
        for (const auto &node: _nodes) {
            if (in_degree.find(node.get()) == in_degree.end()) {
                in_degree[node.get()] = 0;
            }
            for (const auto &next_weak: node->_next_nodes) {
                if (auto next = next_weak.lock()) {
                    in_degree[next.get()]++;
                }
            }
        }

        // 入度为 0 的节点入队
        for (const auto &node: _nodes) {
            if (in_degree[node.get()] == 0) {
                q.push(node);
            }
        }

        while (!q.empty()) {
            auto node = q.front();
            q.pop();
            sorted.push_back(node);

            for (const auto &next_weak: node->_next_nodes) {
                if (auto next = next_weak.lock()) {
                    in_degree[next.get()]--;
                    if (in_degree[next.get()] == 0) {
                        q.push(next);
                    }
                }
            }
        }

        if (sorted.size() != _nodes.size()) {
            // 存在环
            return {};
        }

        return sorted;
    }

    // 前向执行整个图
    bool Graph::forward() {
        auto topo_order = topological_sort();
        if (topo_order.empty()) {
            return false; // 图有环或为空
        }

        return true;
    }

    void Graph::print_topology() const {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto &node: _nodes) {
            printf("Node: %s (op: %d)\n", node->name().c_str(), (int) node->op_type());
            printf("  Inputs: %zu tensors\n", node->_src_tensors_ptr.size());
            printf("  Outputs: %zu tensors\n", node->_dst_tensors_ptr.size());
            printf("  Prev: ");
            for (const auto &prev_weak: node->_prev_nodes) {
                if (auto prev = prev_weak.lock()) {
                    printf("%s ", prev->name().c_str());
                }
            }
            printf("\n  Next: ");
            for (const auto &next_weak: node->_next_nodes) {
                if (auto next = next_weak.lock()) {
                    printf("%s ", next->name().c_str());
                }
            }
            printf("\n");
        }
    }
}
