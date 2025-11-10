//
// Created by nkk on 2025/10/29.
//
#include "taskgraph/include/TaskFlowSchedule.h"
namespace tff::schedule {
    REGISTER_MODULE_OBJECT(HybridScheduler, tff::module::ModuleObject, TASK_GRAPH_FLAG, TASK_GRAPH_TYPE);
    void HybridScheduler::run() {

    }

    void HybridScheduler::wait_until_completion() {
    }
}
