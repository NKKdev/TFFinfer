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
        template<typename F, typename... Args>
        tf::Task add_task(const TaskType &type, const std::string &name, F &&f, Args &&... args);


        // template<typename F, typename... Args>
        // tf::Task add_io_task(const std::string &name, F &&f, Args &&... args);
        //
        // template<typename Kernel, typename... Args>
        // tf::Task add_gpu_task(const std::string &name, Kernel &&kernel, Args &&... args);
        //
        // // 同步点
        // tf::Task add_gpu_wait(const std::string &name, cudaEvent_t event);

        //
        template<typename F, typename... Args>
        tf::Task add_subflow_task(tf::Taskflow &tf, const std::string &name, F &&f, Args &&... args);

        // 提交执行
        void run(const TaskType &type);

        void wait_until_completion(const TaskType &type);

        tf::Taskflow &taskflow(TaskType &type) { return _task_flow[type]; }

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

    // template<typename F, typename... Args>
    // tf::Task HybridScheduler::add_io_task(const std::string &name, F &&f, Args &&... args) {
    //     auto bound = [func = std::forward<F>(f),
    //                 tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
    //         std::apply(func, std::move(tup));
    //     };
    //
    //     return _task_flow.emplace(std::move(bound)).name(name);
    // }
    //
    // template<typename Kernel, typename... Args>
    // tf::Task HybridScheduler::add_gpu_task(const std::string &name, Kernel &&kernel, Args &&... args) {
    //     static size_t stream_idx = 0;
    //     auto &stream = _gpu_streams[stream_idx];
    //     stream_idx = (stream_idx + 1) % _gpu_streams.size();
    //
    //     // 创建事件用于同步
    //     auto &event = _sync_events[stream_idx];
    //
    //     auto gpu_lambda = [this,
    //                 kern = std::forward<Kernel>(kernel),
    //                 tup = std::make_tuple(std::ref(stream), std::forward<Args>(args)...),_event = event,
    //                 _stream = stream]()
    //         -> void {
    //         std::apply(kern, std::move(tup));
    //         cudaEventRecord(_event, _stream);
    //     };
    //
    //     auto task = _task_flow.emplace(std::move(gpu_lambda))
    //             .name(name);
    //
    //     return task;
    // }

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
