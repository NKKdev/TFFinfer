//
// Created by nkk on 2025/11/16.
//

#include "MemManager.h"
#include "global/GlobalDefine.h"
#include "device/DeviceBaseObject.h"

namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMMemManager, tff::module::ModuleObject, WEIGHT_MEM_BUFFER_MANAGER_FLAG,
                           WEIGHT_MEM_BUFFER_MANAGER_FLAG);

    bool LLMMemManager::init(const int &device_id) {
        auto mem_allocator = this->_devices_map[device_id]->get_device_buffer_allocator(device_id);
        this->_mem_buffer_map[device_id] = mem_allocator->allocate_vvm(0);//初始化只获取指针不开辟内存;
        if (this->_mem_buffer_map[device_id] == nullptr) {
            tff::log::Logger::error("device : %d init mem buffer failed!!", device_id);
            return false;
        }
        _is_initialized = true;

        return true;
    }

    int64_t LLMMemManager::allocate_memory(int64_t size, int start, int end, const int &device_id,
                                          const core::memory::MemoryType &type) {
        std::lock_guard<std::mutex> lock(_mutex);

        int64_t offset;
        int64_t actual_size = size;
        if (this->_devices_map[device_id]->get_device_type(device_id) !=
            device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU) {
            actual_size = align_up(size, _alignment);
        }

        if (type == core::memory::MemoryType::TFF_MEM_TYPE_WEIGHT) {
            //权重内存常驻，不参与回收复用;
            offset = _current_offset[device_id];
            _current_offset[device_id] += actual_size;
            _const_mem_offset[device_id].insert(offset);
        } else {
            collect(device_id);

            auto &free_set = _free_set[device_id];
            auto it = free_set.lower_bound({0, actual_size});

            if (it != free_set.end()) {
                const int64_t candidate_offset = it->offset;
                const int64_t candidate_size = it->size;

                if (candidate_size >= actual_size + _alignment) {
                    offset = candidate_offset;
                    int64_t remaining_offset = candidate_offset + actual_size;
                    int64_t remaining_size = candidate_size - actual_size;

                    free_set.erase(it);
                    free_set.insert({remaining_offset, remaining_size});
                } else {
                    offset = candidate_offset;
                    free_set.erase(it);
                }
            } else {
                offset = _current_offset[device_id];
                _current_offset[device_id] += actual_size;
            }
            new_memory_block(device_id, offset, actual_size, end);
        }
        return offset;
    }

    //
    int64_t LLMMemManager::allocate_memory_offset(int64_t size, const int &device_id,
                                          const core::memory::MemoryType &type) {
        std::lock_guard<std::mutex> lock(_mutex);

        int64_t offset;
        int64_t actual_size = size;
        if (this->_devices_map[device_id]->get_device_type(device_id) !=
            device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU) {
            actual_size = align_up(size, _alignment);
        }

        collect(device_id);

        auto &free_set = _free_set[device_id];
        auto it = free_set.lower_bound({0, actual_size});

        if (it != free_set.end()) {
            const int64_t candidate_offset = it->offset;
            const int64_t candidate_size = it->size;

            if (candidate_size >= actual_size + _alignment) {
                offset = candidate_offset;
                int64_t remaining_offset = candidate_offset + actual_size;
                int64_t remaining_size = candidate_size - actual_size;

                free_set.erase(it);
                free_set.insert({remaining_offset, remaining_size});
            } else {
                offset = candidate_offset;
                free_set.erase(it);
            }
        } else {
            offset = _current_offset[device_id];
            _current_offset[device_id] += actual_size;
            if (type == core::memory::MemoryType::TFF_MEM_TYPE_WEIGHT) {
                this->_const_mem_offset[device_id].insert(offset);
            }
            auto mem_allocator =
                this->_devices_map[device_id]->get_device_buffer_allocator(device_id);
            mem_allocator->allocate_vvm(actual_size);
        }
        return offset;
    }

    //
    std::pair<int64_t, void *> LLMMemManager::allocate_memory(int64_t size, const int &device_id,
                                          const core::memory::MemoryType &type) {
        const int64_t offset = allocate_memory_offset(size, device_id, type);
        return std::pair<int64_t, void *>(offset, this->get_ptr_by_offset(device_id, offset));
    }

    void LLMMemManager::release_memory(const int &device_id, const int64_t &offset) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (this->_const_mem_offset[device_id].empty()) {
            return;
        }
        if (this->_const_mem_offset[device_id].find(offset) != this->_const_mem_offset[device_id].end()) {
            return;
        }
        auto iter = this->__async_pending_offset_map[device_id].find(offset);
        if (iter != this->__async_pending_offset_map[device_id].end()) {
            iter->second._ref_count--;
            if (iter->second._ref_count == 0) {
                tff::log::Logger::info("device %d release memory offset %d", device_id, offset);
                if (reclaim_async(device_id, offset)) {
                    this->_memory_heap[device_id].push(iter->second);
                    this->__async_pending_offset_map[device_id].erase(iter);
                }
            }
        }
    }
    //
    void LLMMemManager::aquire_memory(const int &device_id, const size_t &offset,
        const size_t &size,
            std::shared_ptr<core::device::DeviceEvent> &event) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (this->_const_mem_offset[device_id].find(offset) != this->_const_mem_offset[device_id].end()) {
            return;
        }
        auto iter = this->__async_pending_offset_map[device_id].find(offset);
        if (iter != this->__async_pending_offset_map[device_id].end()) {
            iter->second._ref_count++;
        }else {
            this->__async_pending_offset_map[device_id].insert(
                std::make_pair(offset, MemoryBlock(offset, size, 0)));
        }
    }

    bool LLMMemManager::reclaim_async(int device_id, const size_t &offset) {
        auto iter = this->__async_pending_offset_map[device_id].find(offset);
        if (iter != this->__async_pending_offset_map[device_id].end()) {
            auto &block = iter->second;
            if (block._ref_count > 0) {
                return false;
            }
            bool bflag = true;
            for (auto &event: block._pending_events) {
                if (!event->query()) {
                    bflag = false;
                    break;
                }
            }
            return bflag;
        }
        return false;
    }

    void LLMMemManager::reclaim_async(int device_id) {
        auto iter = this->__async_pending_offset_map[device_id].begin();
        for (; iter != this->__async_pending_offset_map[device_id].end();) {
            auto block = iter->second;
            if (block._ref_count > 0) {
                ++iter;
                continue;
            }
            if (reclaim_async(device_id, block._offset)) {
                this->_memory_heap[device_id].push(block);
                iter = this->__async_pending_offset_map[device_id].erase(iter);
            } else {
                ++iter;
            }
        }
    }

    void LLMMemManager::new_memory_block(const int &device_id,
                                         const int64_t &offset, const int64_t &actual_size, const int &end) {
        _memory_heap[device_id].push({offset, actual_size, end});
    }

    //
    void LLMMemManager::collect(const int &device_id) {
        if (_memory_heap[device_id].empty()) {
            reclaim_async(device_id);
        }
        while (!_memory_heap[device_id].empty()) {
            auto &heap = _memory_heap[device_id].top();
            _memory_heap[device_id].pop();
            _free_set[device_id].insert({
                heap._offset,
                heap._size
            });
        }
    }

    //
    void LLMMemManager::pop_pending_block(const int &device_id) {
        auto &pending = _memory_heap[device_id].top();
        auto iter = this->__async_pending_offset_map[device_id].find(pending._offset);
        if (iter != this->__async_pending_offset_map[device_id].end()) {
            this->__async_pending_offset_map[device_id].erase(iter);
        }
    }
    void LLMMemManager::reset(const int &device_id) {
        this->__async_pending_offset_map[device_id].clear();
        this->_current_offset[device_id] = 0;
        this->_free_set[device_id].clear();
        std::priority_queue<MemoryBlock, std::vector<MemoryBlock>, std::greater<>> tmp;
        this->_memory_heap[device_id].swap(tmp);

    }
}
