//
// Created by nkk on 2025/11/10.
//

#ifndef TFFINFER_LLMTASKFLOWMANAGER_H
#define TFFINFER_LLMTASKFLOWMANAGER_H

#include "taskgraph/include/TaskFlowSchedule.h"
#include "ModuleFactory.h"
#include "ModuleObject.h"
#include "graph/Graph.h"
namespace tff::core::runtime {
    class LLMTaskFlowManager : public tff::module::ModuleObject {
    public:
        LLMTaskFlowManager() {
            auto schedule_ptr  =
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(std::string(TASK_GRAPH_FLAG), std::string(TASK_GRAPH_TYPE));
            this->_task_schedule = std::dynamic_pointer_cast<tff::schedule::HybridScheduler>(schedule_ptr);
        };

        ~LLMTaskFlowManager() override = default;
    public:
        void build_task_flow(const std::shared_ptr<tff::core::graph::Graph> &_graph_ptr);
    public:
        std::shared_ptr<tff::schedule::HybridScheduler> _task_schedule;
    };
}

#endif //TFFINFER_LLMTASKFLOWMANAGER_H
