//
// Created by nkk on 2025/11/10.
//

#include "LLMTaskFlowManager.h"

#include "global/ModelGlobalVar.h"
#include "graph/GraphNode.h"

namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMTaskFlowManager, tff::module::ModuleObject, TASK_FLOW_MANAGER_FLAG,
                           tff::core::global::TaskFlowType::TFF_FLOW_LLM);

    bool LLMTaskFlowManager::build_task_schedule(const tff::schedule::TaskType &type,
        const std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
        bool is_fuse) const {
        if (!graph_ptr || !_task_scheduler) {
            return false;
        }

        if (is_fuse) {
            this->fuse_op_node(graph_ptr, graph_ptr->nodes());
        }
        std::unordered_map<std::shared_ptr<tff::core::graph::GraphNode>, tf::Task> node_to_task;
        node_to_task.reserve(graph_ptr->nodes().size());
        for (const auto &node: graph_ptr->nodes()) {
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
            tf::Task task = _task_scheduler->add_task(type, node->name(), std::move(callable), params_ptr);
            node_to_task.emplace(node, std::move(task));
        }

        for (const auto &node: graph_ptr->nodes()) {
            auto current_it = node_to_task.find(node);
            if (current_it == node_to_task.end()) continue;
            for (const auto &pred: node->input_nodes()) {
                auto pred_it = node_to_task.find(pred);
                if (pred_it != node_to_task.end()) {
                    pred_it->second.precede(current_it->second);
                }
            }
        }
        return true;
    }

    void LLMTaskFlowManager::fuse_op_node(const std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                          const std::vector<std::shared_ptr<graph::GraphNode> > &nodes) const {
        for (int i = nodes.size() - 1; i >= 0; --i) {
            auto &node = nodes[i];
            if (node->is_fuse()) {
                //tff::log::Logger::info("op(%s) has fused\n", node->name().c_str());
                continue;
            }
            if (!fuse(graph_ptr, node)) {
                //tff::log::Logger::info("fuse op(%s) do not need fuse!!\n", node->name().c_str());
                continue;
            } else {
                tff::log::Logger::info("op(%s) fused\n", node->name().c_str());
            }
        }
    }

    bool LLMTaskFlowManager::fuse(const std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                  const std::shared_ptr<graph::GraphNode> &current_node) const {
        if (!current_node || current_node->is_fuse()) {
            return false;
        }
        auto match_result = TFF_OP_FUSE_MODEL.find(current_node->op_type());
        if (match_result == TFF_OP_FUSE_MODEL.end()) {
            return false;
        }

        // // 收集所有可融合的前驱或后驱（必须满足：1. 唯一后继是 current_node；2. 同设备；3. 非特殊节点 4. 符合特定融合模式）
        std::vector<std::shared_ptr<graph::GraphNode> > pre_fusible_preds; //待融合的算子集合
        std::vector<std::shared_ptr<graph::GraphNode> > pre_non_fusible_preds; //原始不需要融合的算子集合;
        for (auto &pred: current_node->input_nodes()) {
            if (!pred) {
                continue;
            }

            if (!can_fuse(graph_ptr, current_node, pred)) {
                pre_non_fusible_preds.push_back(pred);
            } else {
                pre_fusible_preds.push_back(pred);
            }
        }

        if (pre_fusible_preds.empty()) {
            return false;
        }
        //
        for (auto &pred: pre_fusible_preds) {
            pred->fuse();
            current_node->remove_src_node(pred);
            for (auto &pre_pred: pred->input_nodes()) {
                current_node->add_src_node(pre_pred);
            }
        }
        return true;
    }

    bool LLMTaskFlowManager::can_fuse(const std::shared_ptr<tff::core::graph::Graph> &graph_ptr,
                                      const std::shared_ptr<graph::GraphNode> &current_node,
                                      std::shared_ptr<graph::GraphNode> &pre_node) const {
        if (pre_node->is_input_node() || pre_node->is_output_node()) {
            return false;
        }

        if (graph_ptr->get_use_count(pre_node) != 1) {
            return false;
        }

        // 设备必须一致
        auto pred_dev = pre_node->device();
        auto curr_dev = current_node->device();
        if (pred_dev.empty() || curr_dev.empty() || pred_dev != curr_dev) {
            return false;
        }

        auto match_result = TFF_OP_FUSE_MODEL.find(current_node->op_type());
        if (match_result == TFF_OP_FUSE_MODEL.end()) {
            return false;
        }
        bool bRet = false;
        if (current_node->op_type() == tff::core::graph::TffOpType::TFF_OP_RMS_NORM) {
            int current_index = graph_ptr->get_node_index(current_node);

            for (int i = 0; i < match_result->second.size(); ++i) {
                if (current_index + i > graph_ptr->nodes().size()) {
                    continue;
                }
                auto &succ_node = graph_ptr->nodes()[current_index + i];
                auto iter = std::find(match_result->second.begin(), match_result->second.end(), succ_node->op_type());
                if (iter != match_result->second.end()) {
                    bRet = true;
                } else {
                    bRet = false;
                }
            }
        }else if (current_node->op_type() == tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT){
            auto iter = std::find(match_result->second.begin(), match_result->second.end(), pre_node->op_type());
            if (iter != match_result->second.end()) {
                bRet = true;
            } else {
                bRet = false;
            }
        }

        return bRet;
    }
}
