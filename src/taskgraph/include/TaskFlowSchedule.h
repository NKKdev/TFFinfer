//
// Created by nkk on 2025/10/29.
//

#ifndef TFFINFER_TASKFLOWSCHEDULE_H
#define TFFINFER_TASKFLOWSCHEDULE_H
#include "device/cuda/cudaInc.h"
#include "TaskflowInc.h"
#include "ModuleFactory.h"
#include "ModuleObject.h"
#include "global/GlobalDefine.h"

namespace tff::schedule {
    enum TaskType {
        TFF_TASK_TYPE_IO,
        TFF_TASK_TYPE_INFER,
    };
    /**
     * @brief HybridScheduler
     * @details
     *
     */
    class HybridScheduler final : public tff::module::ModuleObject {
    public:
        explicit HybridScheduler(const bool use_cuda_graph = false,size_t num_gpu_streams = 1):_use_cuda_graph(use_cuda_graph) {
            cudaStreamCreate(&_capture_stream);
            if (_use_cuda_graph) {
                cudaGraphCreate(&_graph, 0);
            }
        };

        ~HybridScheduler() override {
            if (_graph_exec) {
                cudaGraphExecDestroy(_graph_exec);
            }
            if (_graph) {
                cudaGraphDestroy(_graph);
            }
            if (_capture_stream) {
                cudaStreamDestroy(_capture_stream);
            }
        };

    public:
        /**
         * @brief add_task 添加计算任务
         * @param type 计算任务类型
         * @param name 任务名称
         * @param f 计算任务函数
         * @param args 任务参数
         * @return tf::Task
         */
        template<typename F, typename... Args>
        tf::Task add_task(const TaskType &type, const std::string &name, F &&f, Args &&... args);
        /**
         * @brief add_subflow_task 添加子流任务
         * @param tf 父任务流
         * @param name 任务名称
         * @param f 子流任务函数
         * @param args 子流任务参数
         * @return tf::Task
         */
        template<typename F, typename... Args>
        tf::Task add_subflow_task(tf::Taskflow &tf, const std::string &name, F &&f, Args &&... args);
        /**
         * @brief run 运行任务
         * @param type 任务类型
         */
        void run(const TaskType &type);
        /**
         * @brief wait_until_completion 等待任务完成
         * @param type 任务类型
         */
        void wait_until_completion(const TaskType &type);
        /**
         * @brief taskflow 获取任务流
         * @param type 任务类型
         * @return tf::Taskflow
         */
        tf::Taskflow &taskflow(const TaskType &type) { return _task_flow[type]; }

    private:

        std::unordered_map<TaskType, tf::Executor> _executor;
        std::unordered_map<TaskType, tf::Taskflow> _task_flow;
        std::unordered_map<TaskType, tf::Future<void>> _future;
        cudaStream_t _capture_stream = nullptr;
        cudaGraph_t _graph = nullptr;
        cudaGraphExec_t _graph_exec = nullptr;
        bool _use_cuda_graph = false;


        //todo 多流实现;
        std::vector<cudaStream_t> _gpu_streams;
        std::vector<cudaEvent_t> _sync_events;
    };

    template<typename F, typename... Args>
    tf::Task HybridScheduler::add_task(const TaskType &type, const std::string &name, F &&f, Args &&... args) {
        auto bound = [func = std::forward<F>(f),
                    tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            std::apply(func, tup);
        };
        return _task_flow[type].emplace(std::move(bound)).name(name);
    }
    template<typename F, typename... Args>
    tf::Task HybridScheduler::add_subflow_task(tf::Taskflow &tf, const std::string &name, F &&f, Args &&... args) {
        auto wrapper = [func = std::forward<F>(f),
                    tup = std::make_tuple(std::forward<Args>(args)...)](tf::Subflow &sf) {
            std::apply([&sf, &func](auto &&... unpacked) {
                func(sf, std::forward<decltype(unpacked)>(unpacked)...);
            }, tup);
        };

        auto task = tf.emplace(std::move(wrapper))
                .name(name)
                .sibling();

        return task;
    }
}

#endif //TFFINFER_TASKFLOWSCHEDULE_H
