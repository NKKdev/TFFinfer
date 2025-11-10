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
        TFF_TASK_TYPE_CPU,
        TFF_TASK_TYPE_GPU,
        TFF_TASK_TYPE_IO,
    };

    class HybridScheduler : public tff::module::ModuleObject {
    private:
        tf::Executor _executor;
        tf::Taskflow _taskflow;
        std::vector<cudaStream_t> _gpu_streams;
        std::vector<cudaEvent_t> _sync_events;

    public:
        HybridScheduler(size_t num_gpu_streams = 1){};

        ~HybridScheduler(){};

        template<typename F, typename... Args>
        tf::Task add_cpu_task(const std::string &name, F &&f, Args &&... args);


        template<typename F, typename... Args>
        tf::Task add_io_task(const std::string &name, F &&f, Args &&... args);

        template<typename Kernel, typename... Args>
        tf::Task add_gpu_task(const std::string &name, Kernel &&kernel, Args &&... args);

        // 同步点
        tf::Task add_gpu_wait(const std::string &name, cudaEvent_t event);

        //
        template<typename F, typename... Args>
        tf::Task add_subflow_task(tf::Taskflow &tf, const std::string &name,F &&f, Args &&... args);

        // 提交执行
        void run();

        void wait_until_completion();

        tf::Taskflow &taskflow() { return _taskflow; }
    };

    template<typename F, typename... Args>
    tf::Task HybridScheduler::add_cpu_task(const std::string &name, F &&f, Args &&... args) {
        auto bound = [func = std::forward<F>(f),
                    tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            std::apply(func, std::move(tup)); // 调用 func(args...)
        };
        return _taskflow.emplace(std::move(bound)).name(name).category(TaskType::TFF_TASK_TYPE_CPU);
    }

    template<typename F, typename... Args>
    tf::Task HybridScheduler::add_io_task(const std::string &name, F &&f, Args &&... args) {
        auto bound = [func = std::forward<F>(f),
                    tup = std::make_tuple(std::forward<Args>(args)...)]() mutable {
            std::apply(func, std::move(tup));
        };

        return _taskflow.emplace(std::move(bound)).name(name).category(TaskType::TFF_TASK_TYPE_IO);
    }

    template<typename Kernel, typename... Args>
    tf::Task HybridScheduler::add_gpu_task(const std::string &name, Kernel &&kernel, Args &&... args) {
        static size_t stream_idx = 0;
        auto &stream = _gpu_streams[stream_idx];
        stream_idx = (stream_idx + 1) % _gpu_streams.size();

        // 创建事件用于同步
        auto &event = _sync_events[stream_idx];

        auto gpu_lambda = [this,
                    kern = std::forward<Kernel>(kernel),
                    tup = std::make_tuple(std::ref(stream), std::forward<Args>(args)...),_event = event,
                    _stream = stream]()
            -> void {
            std::apply(kern, std::move(tup));
            cudaEventRecord(_event, _stream);
        };

        auto task = _taskflow.emplace(std::move(gpu_lambda))
                .name(name)
                .category(TaskType::TFF_TASK_TYPE_GPU);

        return task;
    }

    template<typename F, typename ... Args>
    tf::Task HybridScheduler::add_subflow_task(tf::Taskflow &tf, const std::string &name, F &&f, Args &&...args) {
        auto wrapper = [func = std::forward<F>(f),
                    tup = std::make_tuple(std::forward<Args>(args)...)](tf::Subflow& sf) {
            std::apply([&sf, &func](auto&&... unpacked) {
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
