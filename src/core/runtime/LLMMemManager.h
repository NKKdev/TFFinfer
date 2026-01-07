//
// Created by nkk on 2025/11/16.
//

#ifndef TFFINFER_LLMWEIGHTMEMMANAGER_H
#define TFFINFER_LLMWEIGHTMEMMANAGER_H

#include <memory>
#include <mutex>
#include <queue>
#include <vector>
#include <algorithm>
#include "mem/Memory.h"
#include "log/Logger.h"
#include "device/DeviceBaseObject.h"

namespace tff::core::runtime {
    inline size_t align_up(size_t size, size_t alignment) {
        return (size + alignment - 1) & ~(alignment - 1);
    }

    class LLMMemManager final : public tff::module::ModuleObject {
    public:
        explicit LLMMemManager(size_t alignment = 256)
            : _alignment(alignment) {
        }

        ~LLMMemManager() = default;

    public:
        //
        bool init(const int &device_id);

        //
        inline void init_device(
            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > &_devices_map) {
            this->_devices_map = _devices_map;
        }

        int allocate_memory(size_t size, int start, int end, const int &device_id);

        inline size_t get_total_gpu_memory_size(const int &device_id) {
            std::lock_guard<std::mutex> lock(_mutex);
            return _current_offset[device_id];
        }

        inline void bind_memory_pool(const int &device_id, void *base_ptr) {
            this->_mem_buffer_map[device_id] = base_ptr;
        }

        inline void *get_ptr_by_offset(const int &device_id, size_t offset) {
            return static_cast<char *>(this->_mem_buffer_map[device_id]) + offset;
        }

    private:
        struct MemoryBlock {
            size_t _offset;
            int _free_time;

            bool operator>(const MemoryBlock &other) const {
                return _free_time > other._free_time;
            }
        };

        const size_t _alignment;

        mutable std::mutex _mutex;
        std::unordered_map<int, void *> _mem_buffer_map;
        std::unordered_map<int, size_t> _current_offset;
        std::unordered_map<int, std::priority_queue<MemoryBlock, std::vector<MemoryBlock>, std::greater<> > >
        _free_heap;
        std::unordered_map<int, std::vector<size_t> > _free_list;

        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > _devices_map;
    };
}

#endif
