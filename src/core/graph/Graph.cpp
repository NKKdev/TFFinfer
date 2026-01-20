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
        if (node->op_type() == TFF_OP_MEM_REF) {
            return;
        }
        _visited.insert(node);
        _use_counts[node] = 0;

        for (auto &input : node->_input_nodes) {
            if (input != nullptr) {
                visit(input);
                _use_counts[input]++;
            }
        }

        _total_nodes.push_back(node);

        int time_step = static_cast<int>(_total_nodes.size() - 1);
        _exec_time[node] = time_step;

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
        for (auto &node : _total_nodes) {
            _first_use[node] = INF;
            _last_use[node] = -1;
        }

        std::unordered_map<std::shared_ptr<GraphNode>, std::vector<std::shared_ptr<GraphNode>>> consumers;
        for (auto &node : _total_nodes) {
            for (auto &input : node->_input_nodes) {
                if (input) {
                    consumers[input].push_back(node);
                }
            }
        }

        for (auto &node : _total_nodes) {
            if (consumers.find(node) == consumers.end() || consumers[node].empty()) {
                if (_exec_time.find(node) != _exec_time.end()) {
                    int t = _exec_time[node];
                    _first_use[node] = t;
                    _last_use[node] = t;
                }else {
                    tff::log::Logger::error("current node: %s has no exec time");
                    continue;
                }

            } else {
                int min_time = INF, max_time = -1;
                for (auto &consumer : consumers[node]) {
                    if (_exec_time.find(consumer) != _exec_time.end()) {
                        int t = _exec_time[consumer];
                        min_time = std::min(min_time, _exec_time[node]);
                        max_time = std::max(max_time, t);
                    }else {
                        tff::log::Logger::error("current node: %s has no exec time");
                        continue;
                    }

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
