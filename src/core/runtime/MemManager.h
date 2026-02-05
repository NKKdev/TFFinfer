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
        //
        bool init(const int &device_id);

        //
        inline void init_device(
            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > &_devices_map) {
            this->_devices_map = _devices_map;
        }

        //
        int64_t allocate_memory(int64_t size, int start, int end, const int &device_id,
                                const memory::MemoryType &type);

        //
        int64_t allocate_memory_offset(int64_t size, const int &device_id,
                                          const core::memory::MemoryType &type);

        std::pair<int64_t, void *> allocate_memory(int64_t size, const int &device_id,
                                          const core::memory::MemoryType &type);

        void release_memory(const int &device_id, const int64_t &offset);

        //
        void aquire_memory(const int &device_id, const size_t &offset,
                           const size_t &actual_size, std::shared_ptr<device::DeviceEvent> &event);

        inline size_t get_total_memory_size(const int &device_id, const core::memory::MemoryType &type) {
            std::lock_guard<std::mutex> lock(_mutex);
            return _current_offset[device_id];
        }

        inline void *get_ptr_by_offset(const int &device_id, size_t offset) {
            return static_cast<char *>(this->_mem_buffer_map[device_id]) + offset;
        }

        inline bool is_initialized() {
            return _is_initialized;
        };


        //
        void reclaim_async(int device_id);

    private:
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
                : _offset(offset), _size(size), _free_time(free_time), _ref_count(1) {
            }
        };

        struct FreeBlock {
            int64_t offset;
            int64_t size;

            bool operator<(const FreeBlock &other) const {
                if (size != other.size) return size < other.size;
                return offset < other.offset;
            }
        };

    private:
        void new_memory_block(const int &device_id, const int64_t &offset, const int64_t &actual_size, const int &end);

        void collect(const int &device_id);

        //
        void pop_pending_block(const int &device_id);

        bool reclaim_async(int device_id, const size_t &offset);
        //
        void reset(const int &device_id);


        size_t _alignment;
        bool _is_initialized;
        double _buffer_byte_size;
        mutable std::mutex _mutex;
        std::unordered_map<int, void *> _mem_buffer_map;
        std::unordered_map<int, int64_t> _current_offset;
        std::unordered_map<int, int64_t> _pool_size;
        std::unordered_map<int, std::unordered_set<int64_t> > _const_mem_offset;


        std::unordered_map<int, std::priority_queue<MemoryBlock, std::vector<MemoryBlock>, std::greater<> > >
        _memory_heap;
        std::unordered_map<int, std::multiset<FreeBlock> > _free_set;


        std::unordered_map<int, std::unordered_map<int64_t, MemoryBlock> > __async_pending_offset_map;

        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > _devices_map;
    };
}

#endif
