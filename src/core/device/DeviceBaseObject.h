//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_DEVICEBASEOBJECT_H
#define TFFINFER_DEVICEBASEOBJECT_H
#include <memory>
#include "ModuleObject.h"
#include "BaseDefine.h"
#include "MemBufferAllocatorBaseObject.h"
#include "graph/BaseDefine.h"
#include "global/GlobalDefine.h"

namespace tff::core::device {
    /**
     * @brief 设备标签
     * @tparam Type 设备类型
     */
    template<DeviceType Type>
    struct DeviceTag {
        static constexpr DeviceType value = Type;

        static const char *name() {
            if constexpr (Type == TFF_BACKEND_DEVICE_TYPE_CPU) {
                return DEVICE_BACKEND_TYPE_CPU;
            } else if constexpr (Type == TFF_BACKEND_DEVICE_TYPE_GPU) {
                return DEVICE_BACKEND_TYPE_CUDA;
            } else {
                return "UNKNOWN";
            }
        }

        static constexpr bool is_gpu() {
            return Type == TFF_BACKEND_DEVICE_TYPE_GPU;
        }

        static constexpr bool is_cpu() {
            return Type == TFF_BACKEND_DEVICE_TYPE_CPU;
        }

        static constexpr bool supports_async() {
            return is_gpu();
        }

        static constexpr uint32_t default_priority() {
            if constexpr (is_cpu()) {
                return TFF_DEVICE_PRIORITY_CPU;
            } else if constexpr (is_gpu()) {
                return TFF_DEVICE_PRIORITY_GPU;
            } else {
                return TFF_DEVICE_PRIORITY_DEFAULT;
            }
        }

        template<typename T>
        static constexpr bool supports_data_type() {
            if constexpr (is_gpu()) {
                return std::is_same_v<T, float> ||
                       std::is_same_v<T, half> ||
                       std::is_same_v<T, double> ||
                       std::is_same_v<T, int32_t> ||
                       std::is_same_v<T, int64_t> ||
                       std::is_same_v<T, uint8_t>;
            } else if constexpr (is_cpu()) {
                return std::is_same_v<T, float> ||
                       std::is_same_v<T, double> ||
                       std::is_same_v<T, int32_t> ||
                       std::is_same_v<T, int64_t> ||
                       std::is_same_v<T, uint8_t>;
            }
            return false;
        }
    };

    /**
     * @brief 设备流
     */
    class DEEP_TFF_API DeviceStream {
    public:
        /**
         * @brief 析构函数
         */
        virtual ~DeviceStream() = default;

        /**
         * @brief 同步设备流
         */
        virtual void synchronize() = 0;

        /**
         * @brief 获取设备流句柄
         * @return 设备流句柄
         */
        virtual void *get_native_stream() = 0;

        /**
         * @brief 等待设备流完成
         */
        virtual void wait_event(void *event_handle) = 0;

        /**
         * @brief 获取设备流名称
         * @return 设备流名称
         */
        virtual std::string name() = 0;

        /**
         * @brief 设置设备流名称
         * @param name 设备流名称
         */
        virtual void set_name(std::string &name) = 0;

        /**
         * @brief 获取设备ID
         * @return 设备ID
         */
        virtual int device_id() = 0;

        /**
         * @brief 获取设备流是否可用
         * @return 设备流是否可用
         */
        [[nodiscard]] virtual bool is_valid() const = 0;
    };

    /**
     * @brief 设备事件
     */
    class DEEP_TFF_API DeviceEvent {
    public:
        virtual ~DeviceEvent() = default;

    public:
        /**
         * @brief 记录设备流
         * @param stream 设备流
         */
        virtual void record(const std::shared_ptr<DeviceStream> &stream) = 0;

        /**
         * @brief 查询设备流是否完成
         * @return 设备流是否完成
         */
        virtual bool query() = 0;

        /**
         * @brief 获取设备事件句柄
         * @return 设备事件句柄
         */
        virtual void *get_native_event() = 0;

        /**
         * @brief 获取设备事件名称
         * @return 设备事件名称
         */
        [[nodiscard]] virtual bool is_valid() const = 0;

        /**
         * @brief 获取设备事件名称
         * @return 设备事件名称
         */
        virtual std::string name() = 0;

        /**
         * @brief 设置设备事件名称
         * @param name 设备事件名称
         */
        virtual void set_name(std::string &name) = 0;
    };

    /**
     * @brief 设备属性
     */
    class DEEP_TFF_API DeviceBaseObject : public tff::module::ModuleObject {
    public:
        DeviceBaseObject() = default;

        ~DeviceBaseObject() override = default;

    public:
        /**
         * @brief 获取设备列表
         * @param _device_list 设备列表
         */
        virtual void get_device_id(std::vector<int> &_device_list) = 0;

        /**
         *
         * @param _device_id  设备ID
         * @return const char * 设备名称
         */
        virtual const char *get_device_name(size_t _device_id) = 0;

        /**
         *
         * @param _device_id 设备ID
         * @return const char * 设备描述
         */
        virtual const char *get_device_description(size_t _device_id) = 0;

        /**
         *
         * @param _device_id  设备ID
         * @param _free_mem   设备剩余内存
         * @param _total_mem  设备总内存
         */
        virtual void get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) = 0;

        /**
         *
         * @param _device_id  设备ID
         * @return  设备类型
         */
        virtual tff::core::device::DeviceType get_device_type(size_t _device_id) = 0;

        /**
         *
         * @return 设备类型
         */
        virtual DeviceType device_type() = 0;

        /**
         *
         * @param _device_id  设备ID
         * @return 设备类型
         */
        virtual std::string get_device_type_flag(size_t _device_id) = 0;

        /**
         *
         * @param _device_id  设备ID
         * @param _device_props 设备属性
         */
        virtual void get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) = 0;

        /**
         * @brief  初始化设备
         * @param _device_id  设备ID
         *
         */
        virtual void device_init() = 0;

        /**
         * @brief  获取设备内存分配器
         * @param device_id  设备ID
         * @return  设备内存分配器
         */
        virtual std::shared_ptr<MemBufferAllocatorBaseObject> get_device_buffer_allocator(
            const int &device_id) = 0;

        /**
         * @brief  创建设备流
         * @param device_id  设备ID
         * @return  设备流
         */
        virtual std::shared_ptr<DeviceStream> create_stream(int device_id) = 0;

        /**
         * @brief  创建设备事件
         * @param device_id  设备ID
         * @return  设备事件
         */
        virtual std::shared_ptr<DeviceEvent> create_event(int device_id) = 0;

        /**
         * @brief  获取设备时间间隔
         * @param start  设备事件
         * @param stop  设备事件
         * @return  设备时间间隔
         */
        virtual float elapsed_time(
            const std::shared_ptr<DeviceEvent> &start,
            const std::shared_ptr<DeviceEvent> &stop
        ) = 0;

    public:
        uint32_t _sched_priority = TFF_DEVICE_PRIORITY_GPU;
        std::unordered_map<int, std::shared_ptr<MemBufferAllocatorBaseObject> >
        _mem_buffer_allocators;
    };
    /**
     * @brief 设备比较器 优先级比较
     */
    struct DeviceComparator {
        bool operator()(
            const std::shared_ptr<tff::core::device::DeviceBaseObject> &a,
            const std::shared_ptr<tff::core::device::DeviceBaseObject> &b
        ) const {
            if (!a && !b) return false;
            if (!a) return true;
            if (!b) return false;
            return a->_sched_priority > b->_sched_priority;
        }
    };
    /**
     * @brief 获取设备数量
     * @param device_key 设备类型
     * @return 设备数量
     */
    static size_t get_device_size(const std::string &device_key) {
        auto devices = tff::factory::ModuleFactory::instance()->create_shared_list<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG);
        int n_device_num = 0;
        for (const auto &[key, info]: devices) {
            if (tff::factory::ModuleKeyType(device_key) != tff::factory::ModuleKeyType(key)) {
                continue;
            }
            std::vector<int> device_list;
            info.creator()->get_device_id(device_list);
            n_device_num = device_list.size();
        }
        return n_device_num;
    }
    /**
     * @brief 设备标签
     */
    using CPUTag = DeviceTag<TFF_BACKEND_DEVICE_TYPE_CPU>;
    using GPUTag = DeviceTag<TFF_BACKEND_DEVICE_TYPE_GPU>;
}


#endif //TFFINFER_DEVICEBASEOBJECT_H
