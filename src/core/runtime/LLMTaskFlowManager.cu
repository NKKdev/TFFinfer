//
// Created by nkk on 2025/11/10.
//

#include "LLMTaskFlowManager.h"

#include "global/ModelGlobalVar.h"
#include "graph/GraphNode.h"

namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMTaskFlowManager, tff::module::ModuleObject, TASK_FLOW_MANAGER_FLAG,
                           tff::core::global::TaskFlowType::TFF_FLOW_LLM);

    bool LLMTaskFlowManager::build_task_schedule(const std::shared_ptr<tff::core::graph::Graph> &graph_ptr) const {
        if (!graph_ptr || !_task_scheduler) {
            return false;
        }

        auto topo = graph_ptr->topological_sort();
        if (topo.empty()) {
            return false;
        }

        std::unordered_map<std::shared_ptr<tff::core::graph::GraphNode>, tf::Task> node_to_task;
        node_to_task.reserve(topo.size());
        for (const auto& node : topo) {
            if (!node || !node->device()) {
                return false;
            }
            auto callable = node->forward();
            if (!callable) {
                return false;
            }
            auto params_ptr = node->get_params();
            if (!params_ptr) {
                return false;
            }

            tf::Task task = _task_scheduler->add_task(node->name(), std::move(callable), params_ptr);
            node_to_task.emplace(node, std::move(task));
        }

        for (const auto& node : topo) {
            auto current_it = node_to_task.find(node);
            if (current_it == node_to_task.end()) continue;
            for (const auto& pred_weak : node->get_predecessors()) {
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
}
