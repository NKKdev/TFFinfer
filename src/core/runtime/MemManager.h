//
// Created by nkk on 2025/11/16.
//

#ifndef TFFINFER_LLMWEIGHTMEMMANAGER_H
#define TFFINFER_LLMWEIGHTMEMMANAGER_H

#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include <set>
#include <algorithm>
#include <unordered_set>
#include "mem/Memory.h"
#include "mem/BaseDefine.h"
#include "log/Logger.h"
#include "device/DeviceBaseObject.h"

namespace tff::core::runtime {
    inline size_t align_up(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    /**
     * @brief LLM内存管理器
     */
    class LLMMemManager final : public tff::module::ModuleObject {
    public:
        explicit LLMMemManager(size_t alignment = 256)
            : _alignment(alignment), _is_initialized(false), _buffer_byte_size(0) {
        }

        ~LLMMemManager() {
            for (auto &mem_buffer: this->_mem_buffer_map) {
                this->_devices_map[mem_buffer.first]->get_device_buffer_allocator(
                    mem_buffer.first)->release(mem_buffer.second);
            }
            _current_offset.clear();
            this->_free_set.clear();
            this->_devices_map.clear();
        };

    public:
        /**
         * @brief 初始化
         * @param device_id
         * @return
         */
        bool init(const int &device_id);

        /**
         * @brief 初始化设备
         * @param _devices_map
         */
        inline void init_device(
            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > &_devices_map) {
            this->_devices_map = _devices_map;
        }

        /****
         * @brief 申请内存,根据生命周期分配offset(已废弃)
         * @param size 申请内存大小
         * @param start 生命周期开始
         * @param end 生命周期结束
         * @param device_id 设备id
         * @param type 内存类型
         * @return
         */
        int64_t allocate_memory(int64_t size, int start, int end, const int &device_id,
                                const memory::MemoryType &type);

        /**
         * @brief 申请内存
         * @param size 申请内存大小
         * @param device_id 设备id
         * @param type 内存类型
         * @param event 待释放内存的异步事件
         * @return
         */
        int64_t allocate_memory_offset(int64_t size, const int &device_id,
                                       const core::memory::MemoryType &type);

        /**
         * @brief 申请内存
         * @param size 申请内存大小
         * @param device_id 设备id
         * @param type 内存类型
         * @param event 待释放内存的异步事件
         * @return 内存指针
         */
        std::pair<int64_t, void *> allocate_memory(int64_t size, const int &device_id,
                                                   const memory::MemoryType &type,
                                                   const std::shared_ptr<device::DeviceEvent> &event);

        /**
         * @brief 释放内存
         * @param device_id 设备id
         * @param offset 内存偏移
         */
        void release_memory(const int &device_id, const int64_t &offset);

        /**
         * @brief 释放kv缓存
         * @param device_id 设备id
         * @param offset 内存偏移
         */
        void release_kv_cache(const int &device_id, const int64_t &offset);

        /**
         * @brief 重置内存
         * @param device_id 设备id
         */
        void reclaim_memory(const int &device_id);

        /**
         * @brief 获取设备内存引用
         * @param device_id 设备id
         * @param offset 内存偏移
         * @param size 内存大小
         * @param event 待释放内存的异步事件
         */
        void aquire_memory(const int &device_id, const size_t &offset,
                           const size_t &size, const std::shared_ptr<device::DeviceEvent> &event);

        /**
         * @brief 获取设备内存大小
         * @param device_id 设备id
         * @param type 内存类型
         * @return
         */
        inline size_t get_total_memory_size(const int &device_id, const core::memory::MemoryType &type) {
            std::lock_guard<std::mutex> lock(_mutex);
            return _current_offset[device_id];
        }

        /**
         * @brief 获取设备内存指针
         * @param device_id 设备id
         * @param offset 内存偏移
         * @param event 待释放内存的异步事件
         * @return 内存指针
         */
        inline void *get_ptr_by_offset(const int &device_id, size_t offset,
                                       const std::shared_ptr<core::device::DeviceEvent> &event) {
            this->aquire_memory(device_id, offset, this->_mem_buffer_pool[device_id][offset]._size, event);
            return static_cast<char *>(this->_mem_buffer_map[device_id]) + offset;
        }

        /**
         * @brief 是否初始化
         */
        inline bool is_initialized() {
            return _is_initialized;
        };
        /**
         * @brief 释放内存(异步)
         * @param device_id 设备id
         */
        void reclaim_async(int device_id);

    private:
        /**
         * @brief 内存块
         */
        struct MemoryBlock {
            int64_t _offset;
            int64_t _size;
            int _free_time;
            int _ref_count;
            std::vector<std::shared_ptr<core::device::DeviceEvent> > _pending_events;

            bool operator>(const MemoryBlock &other) const {
                return _free_time > other._free_time;
            }

            MemoryBlock(const int64_t offset, const int64_t size, const int free_time)
                : _offset(offset), _size(size), _free_time(free_time), _ref_count(0) {
            }

            MemoryBlock() = default;
        };

        /**
         * @brief 空闲内存块
         */
        struct FreeBlock {
            int64_t _offset;
            int64_t _size;

            bool operator<(const FreeBlock &other) const {
                return _offset < other._offset;
            }
        };

    private:
        /**
         * @brief 更新已使用的内存池
         * @param block 空闲内存块
         * @param device_id 设备id
         * @param type 内存类型
         * @param used_block 已使用的内存块
         */
        void update_used_mem_pool(const FreeBlock &block, const int &device_id, const memory::MemoryType &type,
                                  const FreeBlock &used_block);

        /**
         * @brief 扩充内存池
         * @param device_id 设备id
         * @param offset 内存偏移
         * @param actual_size 实际内存大小
         * @param end 生命周期结束
         */
        void add_memory_pool(const int &device_id, const int64_t &offset, const int64_t &actual_size, const int &end);

        /**
         * @brief 扩充内存堆,用于待使用内存
         * @param device_id 设备id
         * @param offset 内存偏移
         * @param actual_size 实际内存大小
         * @param end 生命周期结束
         */
        void add_memory_heap(const int &device_id, const int64_t &offset, const int64_t &actual_size, const int &end);

        /**
         * @brief 释放内存，释放当前设备所有可释放内存
         * @param device_id 设备id
         */
        void collect(const int &device_id);

        /**
         * @brief 弹出已使用的内存块
         * @param device_id 设备id
         */
        void pop_pending_block(const int &device_id);

        /**
         * @brief 释放内存(同步)
         * @param device_id 设备id
         * @param offset 内存偏移
         * @return
         */
        bool reclaim_async(int device_id, const size_t &offset);

        /**
         * @brief 重置内存
         * @param device_id 设备id
         */
        void reset(const int &device_id);

    public:
        size_t _alignment;
        bool _is_initialized;
        double _buffer_byte_size;
        mutable std::mutex _mutex;
        std::unordered_map<int, void *> _mem_buffer_map;
        std::unordered_map<int, int64_t> _current_offset;
        std::unordered_map<int, int64_t> _pool_size;
        std::unordered_map<int, std::unordered_set<int64_t> > _const_mem_offset;
        std::unordered_map<int, std::unordered_set<int64_t> > _kv_cache_offset;
        std::unordered_map<int, std::unordered_map<int64_t, int64_t> > _temp_buffer_offset;
        std::unordered_map<int, std::unordered_map<int64_t, MemoryBlock> > _mem_buffer_pool;


        std::unordered_map<int, std::priority_queue<MemoryBlock, std::vector<MemoryBlock>, std::greater<> > >
        _memory_heap;
        std::unordered_map<int, std::set<FreeBlock> > _free_set;


        std::unordered_map<int, std::unordered_map<int64_t, MemoryBlock> > __async_pending_offset_map;

        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > _devices_map;
    };
}

#endif
