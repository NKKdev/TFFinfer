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

    class Graph final : public std::enable_shared_from_this<Graph> {
    public:
        Graph() = default;
        ~Graph() = default;

    public:
        void build_graph(std::shared_ptr<GraphNode> output_node);


        const std::vector<std::shared_ptr<GraphNode>>& leafs() const { return _leafs; }
        const std::vector<std::shared_ptr<GraphNode>>& nodes() const { return _nodes; }
        const std::unordered_map<std::shared_ptr<GraphNode>, size_t>& use_counts() const { return _use_counts; }


        void clear();

        bool has_cycle(std::shared_ptr<GraphNode> output_node);
        //
        inline int get_use_count(std::shared_ptr<GraphNode> &node) {
            if (this->_use_counts.find(node) != this->_use_counts.end()) {
                return this->_use_counts[node];
            }else {
                return -1;
            }
        }
        inline int get_node_index(const std::shared_ptr<GraphNode>& node) const {
            auto it = _node_to_index.find(node);
            return (it != _node_to_index.end()) ? static_cast<int>(it->second) : -1;
        }


        inline int get_leaf_index(const std::shared_ptr<GraphNode>& node) const {
            auto it = _leaf_to_index.find(node);
            return (it != _leaf_to_index.end()) ? static_cast<int>(it->second) : -1;

        }


    private:

        void visit(std::shared_ptr<GraphNode> node);


        bool is_leaf_node(const std::shared_ptr<GraphNode>& node) const;

    private:
        std::vector<std::shared_ptr<GraphNode>> _leafs;
        std::vector<std::shared_ptr<GraphNode>> _nodes;
        std::unordered_map<std::shared_ptr<GraphNode>, size_t> _use_counts;
        std::unordered_set<std::shared_ptr<GraphNode>> _visited;

        std::unordered_map<std::shared_ptr<GraphNode>, size_t> _node_to_index;
        std::unordered_map<std::shared_ptr<GraphNode>, size_t> _leaf_to_index;
    };

} // namespace tff::core::graph

#endif // TFFINFER_DAGGRAPH_H
