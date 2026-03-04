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
        /**
         * 创建一个CUDA流
         * @param device_id 设备ID
         */
        CUDAStream(const int &device_id) : _device_id(device_id) {
            CudaSafeCall(cudaSetDevice(_device_id));
            CudaSafeCall(cudaStreamCreate(&_stream));
            _isvalid = true;
        };

        ~CUDAStream() override {
            CudaSafeCall(cudaStreamDestroy(_stream));
        };

    public:
        /**
         * 同步流
         */
        inline void synchronize() override {
            CudaSafeCall(cudaSetDevice(_device_id));
            CudaSafeCall(cudaStreamSynchronize(_stream));
        };
        /**
         * 等待事件
         * @param event_handle 事件句柄
         */
        inline void wait_event(void *event_handle) override {
            if (event_handle == nullptr) {
                return;
            }
            CudaSafeCall(cudaSetDevice(_device_id));
            CudaSafeCall(cudaStreamWaitEvent(this->_stream, static_cast<cudaEvent_t>(event_handle)));
        }

        /**
         * 获取原生流句柄
         * @return
         */
        inline void *get_native_stream() override {
            return static_cast<void *>(_stream);
        };
        /**
         * 是否有效
         * @return
         */
        [[nodiscard]] inline bool is_valid() const override {
            return this->_isvalid;
        };
        /**
         * 获取名称
         * @return
         */
        std::string name() override {
            return this->_name;
        };
        /**
         * 设置名称
         * @param name
         */
        void set_name(std::string &name) override {
            this->_name = name;
        }

        /**
         * 获取设备ID
         * @return
         */
        int device_id() override {
            return this->_device_id;
        }

    protected:
        cudaStream_t _stream = nullptr;
        int _device_id = -1;
        bool _isvalid = false;
        std::string _name;
    };

    //
    class CUDAEvent : public tff::core::device::DeviceEvent {
    public:
        /**
         * 创建一个CUDA事件
         * @param device_id 设备ID
         */
        CUDAEvent(const int &device_id) : _device_id(device_id) {
            CudaSafeCall(cudaEventCreate(&_event));
            _isvalid = true;
        };
        /**
         * 析构函数
         */
        ~CUDAEvent() override {
            CudaSafeCall(cudaEventDestroy(_event));
        };

    public:
        /**
         * 记录事件
         * @param stream 流
         */
        inline void record(const std::shared_ptr<DeviceStream> &stream) override {
            auto cuda_stream = std::dynamic_pointer_cast<DeviceStream>(stream);
            if (!cuda_stream) tff::log::Logger::error("Invalid stream type");
            if (stream->get_native_stream() == nullptr) {
                return;
            }
            CudaSafeCall(cudaEventRecord(_event, static_cast<cudaStream_t>(cuda_stream->get_native_stream())));
        };
        /**
         * 查询事件
         * @return
         */
        bool query() override {
            cudaError_t err = cudaEventQuery(this->_event);
            return err == cudaSuccess;
        }

        /**
         * 获取原生事件句柄
         * @return
         */
        inline void *get_native_event() override {
            return static_cast<void *>(_event);
        };
        /**
         * 是否有效
         * @return
         */
        [[nodiscard]] inline bool is_valid() const override {
            return _isvalid;
        };
        /**
         * 获取名称
         * @return
         */
        std::string name() override {
            return this->_name;
        };
        /**
         * 设置名称
         * @param name
         */
        void set_name(std::string &name) override {
            this->_name = name;
        }

    protected:
        cudaEvent_t _event = nullptr;
        int _device_id = -1;
        bool _isvalid = false;
        std::string _name;
    };

    class DeviceCUDA final : public DeviceBaseObject {
    public:
        /**
         * 构造函数
         */
        DeviceCUDA() {
            this->_sched_priority = TFF_DEVICE_PRIORITY_GPU;
        };

        ~DeviceCUDA() override = default;

    public:
        /**
         * 获取设备间隔时间
         * @param start  事件
         * @param stop   事件
         * @return
         */
        float elapsed_time(const std::shared_ptr<DeviceEvent> &start,
                           const std::shared_ptr<DeviceEvent> &stop) override;

        /**
         * 获取设备ID列表
         * @param _device_list 设备ID列表
         */
        void get_device_id(std::vector<int> &_device_list) override;

        /**
         * 获取设备名称
         * @param _device_id 设备ID
         * @return
         */
        const char *get_device_name(size_t _device_id) override;

        /**
         * 获取设备描述信息
         * @param _device_id  设备ID
         * @return
         */
        const char *get_device_description(size_t _device_id) override;

        /**
         * 获取设备内存信息
         * @param _device_id 设备ID
         * @param _free_mem  设备空闲内存
         * @param _total_mem 设备总内存
         */
        void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) override;

        /**
         * 获取设备类型
         * @param _device_id 设备ID
         * @return
         */
        tff::core::device::DeviceType get_device_type(size_t _device_id) override;

        /**
         * 获取设备类型标志
         * @param _device_id 设备ID
         * @return
         */
        DeviceType device_type() override;

        /**
         * 获取设备类型标志
         * @param _device_id 设备ID
         * @return string 设备类型标志
         */
        std::string get_device_type_flag(size_t _device_id) override;

        /**
         * 获取设备属性
         * @param _device_id 设备ID
         * @param _device_props 设备属性
         */
        void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) override;

        /**
         * 初始化设备
         */
        void device_init() override;

        /**
         * 获取设备内存分配器
         * @param device_id 设备ID
         * @return
         */
        std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> get_device_buffer_allocator(
            const int &device_id) override;

        /**
         * 创建设备流
         * @param device_id 设备ID
         * @return
         */
        std::shared_ptr<DeviceStream> create_stream(int device_id) override;

        /**
         * 创建设备事件
         * @param device_id 设备ID
         * @return
         */
        std::shared_ptr<DeviceEvent> create_event(int device_id) override;
    };
}


#endif //TFFINFER_DEVICECUDA_H
