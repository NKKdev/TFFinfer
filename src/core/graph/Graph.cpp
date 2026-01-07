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
        _total_nodes.push_back(node);
    }

    void Graph::build_graph(std::shared_ptr<GraphNode> output_node) {
        if (!output_node) return;
        clear();

        visit(output_node);
        //
        this->analyze_lifetimes();
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

    //
    void Graph::analyze_lifetimes() {
        const int INF = 1e9;
        for (auto &node: _total_nodes) {
            _first_use[node] = INF;
            _last_use[node] = -1;
        }

        std::unordered_map<std::shared_ptr<GraphNode>, int> exec_time;
        for (int i = 0; i < _total_nodes.size(); ++i) {
            exec_time[_total_nodes[i]] = i;
        }
        std::unordered_map<std::shared_ptr<GraphNode>, std::vector<std::shared_ptr<GraphNode> > > consumers;
        for (auto &node: _total_nodes) {
            for (auto &input: node->_input_nodes) {
                if (input) {
                    consumers[input].push_back(node);
                }
            }
        }
        for (auto &node: _total_nodes) {
            if (consumers.count(node) == 0) {
                _first_use[node] = exec_time[node];
                _last_use[node] = exec_time[node];
            } else {
                int min_time = INF, max_time = -1;
                for (auto &consumer: consumers[node]) {
                    int t = exec_time[consumer];
                    min_time = std::min(min_time, t);
                    max_time = std::max(max_time, t);
                }
                _first_use[node] = min_time;
                _last_use[node] = max_time;
            }
        }
    }

    std::pair<int, int> Graph::get_lifetime(const std::shared_ptr<GraphNode> &node) const {
        auto it1 = _first_use.find(node);
        auto it2 = _last_use.find(node);
        if (it1 != _first_use.end() && it2 != _last_use.end()) {
            return {it1->second, it2->second};
        }
        return {-1, -1};
    }
} // namespace tff::core::graph
