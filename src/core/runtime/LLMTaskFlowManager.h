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
    struct TaskObject {

    };
    class LLMTaskFlowManager : public tff::module::ModuleObject {
    public:
        LLMTaskFlowManager() {
            auto schedule_ptr =
                    tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                        std::string(TASK_GRAPH_FLAG), std::string(TASK_GRAPH_TYPE));
            this->_task_scheduler = std::dynamic_pointer_cast<tff::schedule::HybridScheduler>(schedule_ptr);
        };

        ~LLMTaskFlowManager() override = default;

    public:
        //
        bool build_task_schedule(const std::shared_ptr<tff::core::graph::Graph> &graph_ptr) const;
        //
        inline void run() {
            this->_task_scheduler->run();
        }
        inline std::shared_ptr<tff::schedule::HybridScheduler> get_task_schedule() const {
            return this->_task_scheduler;
        }
        //

    public:
        std::shared_ptr<tff::schedule::HybridScheduler> _task_scheduler;
    };
}

#endif //TFFINFER_LLMTASKFLOWMANAGER_H
