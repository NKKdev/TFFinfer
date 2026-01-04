//
// Created by nkk on 2025/11/10.
//

#include "LLMTaskFlowManager.h"

#include "global/ModelGlobalVar.h"
#include "graph/GraphNode.h"

namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMTaskFlowManager, tff::module::ModuleObject, TASK_FLOW_MANAGER_FLAG,
                           tff::core::global::TaskFlowType::TFF_FLOW_LLM);

    bool LLMTaskFlowManager::build_task_schedule(const bool is_fuse,
                                                 const std::shared_ptr<tff::core::graph::Graph> &graph_ptr) const {
        if (!graph_ptr || !_task_scheduler) {
            return false;
        }

        auto topo = graph_ptr->topological_sort();
        if (topo.empty()) {
            return false;
        }
        if (is_fuse) {
            this->fuse_op_node(topo);
        }
        std::unordered_map<std::shared_ptr<tff::core::graph::GraphNode>, tf::Task> node_to_task;
        node_to_task.reserve(topo.size());
        for (const auto &node: topo) {
            //tff::log::Logger::info("layer node: %s build op callback\n", node->name().c_str());
            if (!node || node->device().empty() || node->is_fuse()) {
                continue;
            }

            auto callable = node->forward();
            if (!callable) {
                tff::log::Logger::error("layer node: %s has not op callback!", node->name().c_str());
                continue;
            }
            auto params_ptr = node->get_params();
            if (!params_ptr) {
                continue;
            }
            tf::Task task = _task_scheduler->add_task(node->name(), std::move(callable), params_ptr);
            node_to_task.emplace(node, std::move(task));
        }

        for (const auto &node: topo) {
            auto current_it = node_to_task.find(node);
            if (current_it == node_to_task.end()) continue;
            for (const auto &pred_weak: node->get_predecessors()) {
                if (auto pred = pred_weak.lock()) {
                    auto pred_it = node_to_task.find(pred);
                    if (pred_it != node_to_task.end()) {
                        pred_it->second.precede(current_it->second);
                    }
                }
            }
        }
        return true;
    }

    void LLMTaskFlowManager::fuse_op_node(std::vector<std::shared_ptr<graph::GraphNode> > &nodes) const {
        for (int i = nodes.size() - 1; i >= 0; --i) {
            auto &node = nodes[i];
            if (node->is_fuse()) {
                tff::log::Logger::info("op(%s) has fused\n", node->name().c_str());
                continue;
            }
            if (!fuse(node)) {
                tff::log::Logger::info("fuse op(%s) do not need fuse!!\n", node->name().c_str());
                continue;
            } else {
                tff::log::Logger::info("op(%s) fused\n", node->name().c_str());
            }
        }
    }

    bool LLMTaskFlowManager::fuse(std::shared_ptr<graph::GraphNode> &current_node) const{
        if (!current_node || current_node->is_fuse()) {
            return false;
        }
        auto match_result = TFF_OP_FUSE_MODEL.find(current_node->op_type());
        if (match_result == TFF_OP_FUSE_MODEL.end()) {
            return false;
        }

        // 收集所有可融合的前驱或后驱（必须满足：1. 唯一后继是 current_node；2. 同设备；3. 非特殊节点 4. 符合特定融合模式）
        std::vector<std::shared_ptr<graph::GraphNode> > pre_fusible_preds;//待融合的算子集合
        std::vector<std::shared_ptr<graph::GraphNode> > pre_non_fusible_preds;//原始不需要融合的算子集合;
        for (const auto &weak_pred: current_node->get_predecessors()) {
            auto pred = weak_pred.lock();
            if (!pred) {
                continue;
            }

            if (!can_fuse(current_node, pred)) {
                pre_non_fusible_preds.push_back(pred);
            }else {
                pre_fusible_preds.push_back(pred);
            }
        }
        std::vector<std::shared_ptr<graph::GraphNode> > succ_fusible_preds;//待融合的算子集合
        std::vector<std::shared_ptr<graph::GraphNode> > succ_non_fusible_preds;//原始不需要融合的算子集合;
        //
        for (const auto &weak_pred: current_node->get_successors()) {
            auto pred = weak_pred.lock();
            if (!pred) {
                continue;
            }

            if (!can_fuse(current_node, pred)) {
                succ_non_fusible_preds.push_back(pred);
            }else {
                succ_fusible_preds.push_back(pred);
            }
        }
        //
        if (succ_fusible_preds.empty() && pre_fusible_preds.empty()) {
            return false;
        }

        for (auto &pred: pre_fusible_preds) {
            for (const auto &weak_grand: pred->get_predecessors()) {
                auto grand = weak_grand.lock();
                if (!grand) continue;

                grand->add_successors(current_node);
                current_node->add_predecessors(weak_grand);
            }

            pred->fuse();
            pred->erase_successors(current_node);
            current_node->erase_predecessors(pred);
        }

        for (auto &pred: succ_fusible_preds) {
            for (const auto &weak_grand: pred->get_successors()) {
                auto grand = weak_grand.lock();
                if (!grand) continue;

                grand->add_predecessors(current_node);
                current_node->add_successors(weak_grand);
            }

            pred->fuse();
            pred->erase_predecessors(current_node);
            current_node->erase_successors(pred);
        }
        return true;
    }
    bool LLMTaskFlowManager::can_fuse(std::shared_ptr<graph::GraphNode> &current_node,
        std::shared_ptr<graph::GraphNode> &pre_node) const {
        if (pre_node->is_input_node() || pre_node->is_output_node()) {
            return false;
        }

        auto successors = pre_node->get_successors();
        if (successors.size() != 1) {
            return false;
        }
        auto succ = successors[0].lock();
        if (!succ || succ != current_node) {
            return false;
        }

        // 设备必须一致
        auto pred_dev = pre_node->device();
        auto curr_dev = current_node->device();
        if (pred_dev.empty() || curr_dev.empty()) {// todo 20260104;
            return false;
        }
        auto match_result = TFF_OP_FUSE_MODEL.find(current_node->op_type());
        if (match_result == TFF_OP_FUSE_MODEL.end()) {
            return false;
        }
        auto iter = std::find(match_result->second.begin(), match_result->second.end(), pre_node->op_type());
        if (iter != match_result->second.end()) {
            return true;
        }else {
            return false;
        }
        return true;
    }
}
