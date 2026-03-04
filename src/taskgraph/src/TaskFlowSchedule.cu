//
// Created by nkk on 2025/10/29.
//
#include "taskgraph/include/TaskFlowSchedule.h"
namespace tff::schedule {
    REGISTER_MODULE_OBJECT(HybridScheduler, tff::module::ModuleObject, TASK_GRAPH_FLAG, TASK_GRAPH_TYPE);
    void HybridScheduler::run(const TaskType &type) {
        if (_use_cuda_graph) {
            throw std::runtime_error("CUDA Graph mode not fully implemented yet.");
        }
        _future[type] = _executor[type].run(_task_flow[type]);
        wait_until_completion(type);
#ifdef _DEBUG
        std::string dump_str = _task_flow[type].dump();
        const char* graph_dump_path = "task_flow_graph.json";
        std::ofstream file(graph_dump_path);
        if (file.is_open()) {
            file << dump_str;
            file.close();
            //tff::log::Logger::info("Task flow graph dumped to: %s\n", graph_dump_path);
        } else {
            tff::log::Logger::warning("Failed to open file for dumping task graph: %s\n", graph_dump_path);
        }
#endif
    }

    void HybridScheduler::wait_until_completion(const TaskType &type) {
        if (_future[type].valid()) {
            _future[type].get();
        }
    }
}
