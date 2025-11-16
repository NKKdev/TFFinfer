//
// Created by nkk on 2025/10/29.
//
#include "taskgraph/include/TaskFlowSchedule.h"
namespace tff::schedule {
    REGISTER_MODULE_OBJECT(HybridScheduler, tff::module::ModuleObject, TASK_GRAPH_FLAG, TASK_GRAPH_TYPE);
    void HybridScheduler::run() {
        if (_use_cuda_graph) {
            throw std::runtime_error("CUDA Graph mode not fully implemented yet.");
        }
        _future = _executor.run(_task_flow);
    }

    void HybridScheduler::wait_until_completion() {
        if (_future.valid()) {
            _future.get();
        }
    }
}
