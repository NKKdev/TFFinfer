//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_DEVICECUDA_H
#define TFFINFER_DEVICECUDA_H
#include "../DeviceBaseObject.h"
#include "ModuleFactory.h"
#include "global/GlobalDefine.h"
#include <vector>
#include "cudaInc.h"

namespace tff::core::device::cuda {
    class CUDAStream : public tff::core::device::DeviceStream {
    public:
        CUDAStream(const int &device_id) : _device_id(device_id) {
            CudaSafeCall(cudaSetDevice(_device_id));
            CudaSafeCall(cudaStreamCreate(&_stream));
            _isvalid = true;
        };

        ~CUDAStream() override {
            CudaSafeCall(cudaStreamDestroy(_stream));
        };

        inline void synchronize() override {
            CudaSafeCall(cudaSetDevice(_device_id));
            CudaSafeCall(cudaStreamSynchronize(_stream));
        };
        inline void wait_event(void *event_handle) override{
            CudaSafeCall(cudaStreamWaitEvent(this->_stream, static_cast<cudaEvent_t>(event_handle)));
        }
        inline void *get_native_stream() override {
            return static_cast<void *>(_stream);
        };

        [[nodiscard]] inline bool is_valid() const override {
            return this->_isvalid;
        };

    protected:
        cudaStream_t _stream = nullptr;
        int _device_id = -1;
        bool _isvalid = false;
    };

    //
    class CUDAEvent : public tff::core::device::DeviceEvent {
    public:
        CUDAEvent(const int &device_id) : _device_id(device_id) {
            CudaSafeCall(cudaSetDevice(_device_id));
            CudaSafeCall(cudaEventCreate(&_event));
            _isvalid = true;
        };

        ~CUDAEvent() override {
            CudaSafeCall(cudaEventDestroy(_event));
        };

    public:
        inline void record(const std::shared_ptr<DeviceStream> &stream) override {
            auto cuda_stream = std::dynamic_pointer_cast<DeviceStream>(stream);
            if (!cuda_stream) tff::log::Logger::error("Invalid stream type");
            cudaSetDevice(_device_id);
            cudaEventRecord(_event, static_cast<cudaStream_t>(cuda_stream->get_native_stream()));
        };


        inline void *get_native_event() override {
            return static_cast<void *>(_event);
        };

        // 是否有效
        [[nodiscard]] inline bool is_valid() const override {
            return _isvalid;
        };


    protected:
        cudaEvent_t _event = nullptr;
        int _device_id = -1;
        bool _isvalid = false;
    };

    class DeviceCUDA final : public DeviceBaseObject {
    public:
        DeviceCUDA() {
            this->_sched_priority = TFF_DEVICE_PRIORITY_GPU;
        };

        ~DeviceCUDA() override = default;

    public:
        float elapsed_time(const std::shared_ptr<DeviceEvent> &start, const std::shared_ptr<DeviceEvent> &stop) override;

        void get_device_id(std::vector<int> &_device_list) override;

        const char *get_device_name(size_t _device_id) override;

        const char *get_device_description(size_t _device_id) override;

        void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) override;

        tff::core::device::DeviceType get_device_type(size_t _device_id) override;

        tff::core::device::DeviceType device_type() override;

        std::string get_device_type_flag(size_t _device_id) override;

        void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) override;

        void device_init() override;

        std::shared_ptr<tff::core::memory::MemBufferAllocatorBaseObject> get_device_buffer_allocator(
            const int &device_id) override;

        std::function<tff::kernel::base::OP_CALLBACK_TYPE> get_op_func(
            const tff::core::graph::TffOpType &op_type, const tff::core::memory::DataType &data_type) override;

        std::shared_ptr<DeviceStream> create_stream(int device_id) override;

        std::shared_ptr<DeviceEvent> create_event(int device_id) override;
    };
}


#endif //TFFINFER_DEVICECUDA_H
