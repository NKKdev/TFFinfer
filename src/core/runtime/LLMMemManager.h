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
#include "mem/Memory.h"
#include "log/Logger.h"
#include "device/DeviceBaseObject.h"

namespace tff::core::runtime {
    enum class MemoryType {
        WEIGHT,      // 持久权重内存，不可回收
        ACTIVATION   // 临时激活内存，可复用
    };

    inline size_t align_up(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    class LLMMemManager final : public tff::module::ModuleObject {
    public:
        explicit LLMMemManager(size_t alignment = 256)
            : _alignment(alignment),_is_initialized(false),_buffer_byte_size(0) {
        }

        ~LLMMemManager() {
            auto iter = this->_mem_buffer_map.begin()->second.begin();
            for (; iter != this->_mem_buffer_map.begin()->second.end(); ++iter) {
                this->_devices_map[iter->first]->get_device_buffer_allocator(iter->first)->release(iter->second);
            }
            _current_offset.clear();
            this->_free_set.clear();
            this->_free_heap.clear();
            this->_devices_map.clear();
        };

    public:
        //
        bool init(const int &device_id, const MemoryType &type);

        //
        inline void init_device(
            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > &_devices_map) {
            this->_devices_map = _devices_map;
        }

        size_t allocate_memory(size_t size, int start, int end, const int &device_id, const MemoryType &type);

        inline size_t get_total_memory_size(const int &device_id, const MemoryType &type) {
            std::lock_guard<std::mutex> lock(_mutex);
            return _current_offset[type][device_id];
        }

        inline void *get_ptr_by_offset(const int &device_id, size_t offset, const MemoryType &type) {
            return static_cast<char *>(this->_mem_buffer_map[type][device_id]) + offset;
        }
        inline bool is_initialized(const MemoryType &type) {
            return _is_initialized[type];
        };


    private:
        struct MemoryBlock {
            size_t _offset;
            size_t _size;
            int _free_time;

            bool operator>(const MemoryBlock &other) const {
                return _free_time > other._free_time;
            }
        };
        struct FreeBlock {
            size_t offset;
            size_t size;
            bool operator<(const FreeBlock& other) const {
                if (size != other.size) return size < other.size;
                return offset < other.offset;
            }
        };

        size_t _alignment;
        std::unordered_map<MemoryType, bool> _is_initialized;
        double _buffer_byte_size;
        mutable std::mutex _mutex;
        std::unordered_map<MemoryType, std::unordered_map<int, void *>> _mem_buffer_map;
        std::unordered_map<MemoryType, std::unordered_map<int, size_t>> _current_offset;

        std::unordered_map<int, std::priority_queue<MemoryBlock, std::vector<MemoryBlock>, std::greater<> > >
        _free_heap;

        std::unordered_map<MemoryType, std::unordered_map<int, std::multiset<FreeBlock>>> _free_set;

        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > _devices_map;
    };
}

#endif
