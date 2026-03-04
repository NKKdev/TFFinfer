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
    /**
     * @brief LLM任务调度管理器
     */
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
        /**
         * @brief 构建任务调度
         * @param type 任务类型
         * @param graph_ptr 图
         * @return 是否成功
         */
        bool build_task_schedule(const schedule::TaskType &type, const std::shared_ptr<graph::Graph> &graph_ptr) const;
        /**
         * @brief 运行任务调度
         * @param type 运行任务类型
         */
        inline void run(const tff::schedule::TaskType &type) {
            this->_task_scheduler->run(type);
        }
        /**
         * @brief 获取调度器
         * @return 调度器
         */
        inline std::shared_ptr<tff::schedule::HybridScheduler> get_task_schedule() const {
            return this->_task_scheduler;
        }
    protected:
        std::shared_ptr<tff::schedule::HybridScheduler> _task_scheduler;
    };
}

#endif //TFFINFER_LLMTASKFLOWMANAGER_H
