//
// Created by nkk on 2025/11/10.
//

#include "TaskFlowManager.h"

#include "global/ModelGlobalVar.h"
#include "graph/GraphNode.h"
#include "include/Builder.h"
#include "runtime/MemOptStrategy.h"
namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMTaskFlowManager, tff::module::ModuleObject, TASK_FLOW_MANAGER_FLAG,
                           tff::core::global::TaskFlowType::TFF_FLOW_LLM);

    bool LLMTaskFlowManager::build_task_schedule(const tff::schedule::TaskType &type,
        const std::shared_ptr<tff::core::graph::Graph> &graph_ptr) const {
        if (!graph_ptr || !_task_scheduler) {
            return false;
        }
        _task_scheduler->taskflow(type).clear();

        std::unordered_map<std::shared_ptr<tff::core::graph::GraphNode>, tf::Task> node_to_task;
        node_to_task.reserve(graph_ptr->nodes().size());
        for (const auto &node: graph_ptr->total_nodes()) {
            //tff::log::Logger::info("layer node: %s build op callback\n", node->name().c_str());
            if (!node) {
                continue;
            }
            if (node->op_type() == graph::TFF_OP_ATTN_MASK) {
                tff::log::Logger::info("layer node: %s build op callback\n", node->name().c_str());
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

        for (const auto &node: graph_ptr->total_nodes()) {
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
}
