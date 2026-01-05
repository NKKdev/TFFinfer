#include "Graph.h"
#include <cassert>

namespace tff::core::graph {
    void Graph::clear() {
        _leafs.clear();
        _nodes.clear();
        _use_counts.clear();
        _visited.clear();
        _node_to_index.clear();
        _leaf_to_index.clear();
    }
    bool Graph::is_leaf_node(const std::shared_ptr<GraphNode> &node) const {
        return node->is_leaf();
    }

    void Graph::visit(std::shared_ptr<GraphNode> node) {
        if (!node) return;
        if (_visited.find(node) != _visited.end()) {
            return;
        }

        _visited.insert(node);
        _use_counts[node] = 0;


        for (auto &input: node->_input_nodes) {
            if (input != nullptr) {
                visit(input);
                _use_counts[input]++;
            }
        }

        if (is_leaf_node(node)) {
            size_t idx = _leafs.size();
            _leafs.push_back(node);
            _leaf_to_index[node] = idx;
        } else {
            size_t idx = _nodes.size();
            _nodes.push_back(node);
            _node_to_index[node] = idx;

        }
    }

    void Graph::build_graph(std::shared_ptr<GraphNode> output_node) {
        if (!output_node) return;
        clear();

        visit(output_node);
    }

    bool Graph::has_cycle(std::shared_ptr<GraphNode> output_node) {
        std::unordered_set<std::shared_ptr<GraphNode> > visited;
        std::unordered_set<std::shared_ptr<GraphNode> > visiting;

        std::function<bool(std::shared_ptr<GraphNode>)> dfs = [&](std::shared_ptr<GraphNode> node) -> bool {
            if (!node) return false;
            if (visiting.count(node)) return true; // back edge → cycle
            if (visited.count(node)) return false;

            visiting.insert(node);
            for (auto &in: node->_input_nodes) {
                if (dfs(in)) return true;
            }
            visiting.erase(node);
            visited.insert(node);
            return false;
        };

        return dfs(output_node);
    }

} // namespace tff::core::graph
