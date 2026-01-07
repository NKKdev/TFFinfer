//
// Created by nkk on 2025/11/10.
//

#ifndef TFFINFER_LLMTASKFLOWMANAGER_H
#define TFFINFER_LLMTASKFLOWMANAGER_H

#include "taskgraph/include/TaskFlowSchedule.h"
#include "ModuleFactory.h"
#include "ModuleObject.h"
#include "graph/Graph.h"
#include "LLMMemManager.h"
namespace tff::core::runtime {
    struct TaskObject {

    };
    class LLMTaskFlowManager : public tff::module::ModuleObject {
    public:
        LLMTaskFlowManager() {
            auto schedule_ptr =
                    tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                        std::string(TASK_GRAPH_FLAG), std::string(TASK_GRAPH_TYPE));
            this->_task_scheduler = std::dynamic_pointer_cast<tff::schedule::HybridScheduler>(schedule_ptr);
            this->_weight_mem_manager_ptr = std::dynamic_pointer_cast<tff::core::runtime::LLMMemManager>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    WEIGHT_MEM_BUFFER_MANAGER_FLAG,
                    tff::factory::ModuleKeyType(WEIGHT_MEM_BUFFER_MANAGER_FLAG)));
        };

        ~LLMTaskFlowManager() override = default;

    public:
        //
        bool build_task_schedule(bool is_fuse, const std::shared_ptr<graph::Graph> &graph_ptr) const;
        //
        inline void run() {
            this->_task_scheduler->run();
        }
        inline std::shared_ptr<tff::schedule::HybridScheduler> get_task_schedule() const {
            return this->_task_scheduler;
        }
    protected:
        void fuse_op_node(const std::shared_ptr<graph::Graph> &graph_ptr, const std::vector<std::shared_ptr<graph::GraphNode>> &nodes) const;
        bool fuse(const std::shared_ptr<graph::Graph> &graph_ptr, const std::shared_ptr<graph::GraphNode> &current_node) const;
        //
        bool can_fuse(const std::shared_ptr<graph::Graph> &graph_ptr, const std::shared_ptr<graph::GraphNode> &current_node, std::shared_ptr<
                      graph::GraphNode> &pre_node) const;

    protected:
        std::shared_ptr<tff::schedule::HybridScheduler> _task_scheduler;
        std::shared_ptr<tff::core::runtime::LLMMemManager> _weight_mem_manager_ptr;
    };
}

#endif //TFFINFER_LLMTASKFLOWMANAGER_H
