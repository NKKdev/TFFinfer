//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_DEVICECPU_H
#define TFFINFER_DEVICECPU_H
#include "device/DeviceBaseObject.h"
#include "global/GlobalDefine.h"

namespace tff::core::device::cpu {
        class CPUStream : public tff::core::device::DeviceStream {
    public:
        CPUStream(const int &device_id) : _device_id(device_id) {
            _isvalid = true;
        };

        ~CPUStream() override {

        };

        inline void synchronize() override {

        };
        inline void wait_event(void *event_handle) override{

        }
        inline void *get_native_stream() override {
            return nullptr;
        };

        [[nodiscard]] inline bool is_valid() const override {
            return this->_isvalid;
        };

        std::string name() override {
            return this->_name;
        };

        void set_name(std::string &name) override {
            this->_name = name;
        }
        //
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
    class CPUEvent : public tff::core::device::DeviceEvent {
    public:
        CPUEvent(const int &device_id) : _device_id(device_id) {
            _isvalid = true;
        };

        ~CPUEvent() override {
        };

    public:
        inline void record(const std::shared_ptr<DeviceStream> &stream) override {
        };
        bool query() override {
           return true;
        }

        inline void *get_native_event() override {
            return nullptr;
        };

        // 是否有效
        [[nodiscard]] inline bool is_valid() const override {
            return _isvalid;
        };

        std::string name() override {
            return this->_name;
        };

        void set_name(std::string &name) override {
            this->_name = name;
        }


    protected:
        cudaEvent_t _event = nullptr;
        int _device_id = -1;
        bool _isvalid = false;
        std::string _name;
    };
    class DeviceCPU : public DeviceBaseObject {
    public:
        DeviceCPU() {
            this->_sched_priority = TFF_DEVICE_PRIORITY_CPU;
        };

        ~DeviceCPU() override = default;

    public:
        float elapsed_time(const std::shared_ptr<DeviceEvent> &start, const std::shared_ptr<DeviceEvent> &stop) override;
        void get_device_id(std::vector<int> &_device_list) override;

        const char *get_device_name(size_t _device_id) override;

        const char *get_device_description(size_t _device_id) override;

        void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) override;

        tff::core::device::DeviceType get_device_type(size_t _device_id) override;

        DeviceType device_type();

        std::string get_device_type_flag(size_t _device_id);

        void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) override;

        void device_init() override;

        std::shared_ptr<MemBufferAllocatorBaseObject> get_device_buffer_allocator(const int &device_id) override;

        std::shared_ptr<DeviceStream> create_stream(int device_id) override;

        std::shared_ptr<DeviceEvent> create_event(int device_id) override;
    };



}

#endif //TFFINFER_DEVICECPU_H
