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
        this->_mem_buffer_map[device_id] = mem_allocator->allocate_vvm(0); //初始化只获取指针不开辟内存;
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

        if (type == core::memory::MemoryType::TFF_MEM_TYPE_RESIDENT) {
            //权重内存常驻，不参与回收复用;
            offset = _current_offset[device_id];
            _current_offset[device_id] += actual_size;
            _const_mem_offset[device_id].insert(offset);
        } else {
            collect(device_id);

            auto &free_set = _free_set[device_id];
            auto it = free_set.lower_bound({0, actual_size});

            if (it != free_set.end()) {
                const int64_t candidate_offset = it->_offset;
                const int64_t candidate_size = it->_size;

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
            add_memory_pool(device_id, offset, actual_size, end);
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


        if (type == memory::MemoryType::TFF_MEM_TYPE_RESIDENT) {
            offset = _current_offset[device_id];
            _current_offset[device_id] += actual_size;
            this->_const_mem_offset[device_id].insert(offset);
            auto mem_allocator =
                    this->_devices_map[device_id]->get_device_buffer_allocator(device_id);
            mem_allocator->allocate_vvm(actual_size);
            add_memory_pool(device_id, offset, actual_size, 0);
        } else {
            collect(device_id);

            auto &free_set = _free_set[device_id];
            //auto it = free_set.lower_bound({0, actual_size});
            //first fit
            auto it = _free_set[device_id].end();
            for (auto fit = _free_set[device_id].begin(); fit != _free_set[device_id].end(); ++fit) {
                if (fit->_size >= actual_size) {
                    it = fit;
                    break;
                }
            }

            if (it != free_set.end()) {
                const int64_t candidate_offset = it->_offset;
                const int64_t candidate_size = it->_size;

                if (candidate_size >= actual_size + _alignment) {
                    offset = candidate_offset;
                    int64_t remaining_offset = candidate_offset + actual_size;
                    int64_t remaining_size = candidate_size - actual_size;

                    update_used_mem_pool(*it, device_id, type, FreeBlock{offset, actual_size});
                    free_set.erase(it);
                    free_set.insert({remaining_offset, remaining_size});
                } else {
                    offset = candidate_offset;
                    update_used_mem_pool(*it, device_id, type, FreeBlock{offset, actual_size});
                    free_set.erase(it);
                }
            } else {
                offset = _current_offset[device_id];
                _current_offset[device_id] += actual_size;

                if (type == core::memory::MemoryType::TFF_MEM_TYPE_KV_CACHE) {
                    this->_kv_cache_offset[device_id].insert(offset);
                } else {
                    this->_temp_buffer_offset[device_id].insert(std::make_pair(offset, actual_size));
                }
                auto mem_allocator =
                        this->_devices_map[device_id]->get_device_buffer_allocator(device_id);
                mem_allocator->allocate_vvm(actual_size);
                add_memory_pool(device_id, offset, actual_size, 0);
            }
        }


        return offset;
    }

    //
    std::pair<int64_t, void *> LLMMemManager::allocate_memory(int64_t size, const int &device_id,
                                                              const core::memory::MemoryType &type,
                                                              const std::shared_ptr<core::device::DeviceEvent> &event) {
        const int64_t offset = allocate_memory_offset(size, device_id, type);
        // tff::log::Logger::info(
        //         "device_id: %d,memory type: %d, allocate offset: %lld end offset : %lld",device_id, type, offset,
        //         offset + size);
        return std::pair<int64_t, void *>(offset, this->get_ptr_by_offset(device_id, offset, event));
    }

    void LLMMemManager::release_memory(const int &device_id, const int64_t &offset) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (this->_const_mem_offset[device_id].contains(offset) ||
            this->_kv_cache_offset[device_id].contains(offset)) {
            return;
        }
        if (this->__async_pending_offset_map[device_id].contains(offset)) {
            auto &block = this->__async_pending_offset_map[device_id][offset];
            block._ref_count--;
            if (block._ref_count == 0) {
                //tff::log::Logger::info("device %d release memory offset %lld", device_id, offset);
                if (reclaim_async(device_id, offset)) {
                    add_memory_heap(device_id, offset, block._size, block._free_time);
                    this->__async_pending_offset_map[device_id].erase(
                        this->__async_pending_offset_map[device_id].find(offset));
                }
            }
        } else {
            if (this->_mem_buffer_pool[device_id].contains(offset)) {
                add_memory_heap(device_id,  this->_mem_buffer_pool[device_id][offset]._offset,
                    this->_mem_buffer_pool[device_id][offset]._size,
                    this->_mem_buffer_pool[device_id][offset]._free_time);
            }
        }
    }

    //
    void LLMMemManager::release_kv_cache(const int &device_id, const int64_t &offset) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (!this->_kv_cache_offset[device_id].contains(offset)) {
            return;
        }
        if (this->__async_pending_offset_map[device_id].contains(offset)) {
            auto &block = this->__async_pending_offset_map[device_id][offset];
            block._ref_count--;
            if (block._ref_count == 0) {
                //tff::log::Logger::info("device %d release memory offset %lld", device_id, offset);
                if (reclaim_async(device_id, offset)) {
                    add_memory_heap(device_id, offset, block._size, block._free_time);
                    this->__async_pending_offset_map[device_id].erase(
                        this->__async_pending_offset_map[device_id].find(offset));
                }
            }
        } else {
            if (this->_mem_buffer_pool[device_id].contains(offset)) {
                add_memory_heap(device_id,  this->_mem_buffer_pool[device_id][offset]._offset,
                    this->_mem_buffer_pool[device_id][offset]._size,
                    this->_mem_buffer_pool[device_id][offset]._free_time);
            }
        }
    }

    void LLMMemManager::reclaim_memory(const int &device_id) {
        std::lock_guard<std::mutex> lock(_mutex);
        for (const auto &[offset, size]: this->_temp_buffer_offset[device_id]) {
            //tff::log::Logger::info("device %d reclaim memory offset %lld size: %lld", device_id, offset, size);
            add_memory_heap(device_id, offset, size, 0);
        }
        this->_temp_buffer_offset[device_id].clear();
    }

    //
    void LLMMemManager::aquire_memory(const int &device_id, const size_t &offset,
                                      const size_t &size,
                                      const std::shared_ptr<core::device::DeviceEvent> &event) {
        std::lock_guard<std::mutex> lock(_mutex);
        if (this->_const_mem_offset[device_id].contains(offset)) {
            return;
        }
        if (this->__async_pending_offset_map[device_id].contains(offset)) {
            this->__async_pending_offset_map[device_id][offset]._ref_count++;
            // tff::log::Logger::info("device %d aquire memory offset %lld size: %lld, _ref_count: %d",
            //     device_id, offset, size, this->__async_pending_offset_map[device_id][offset]._ref_count);
        } else {
            this->__async_pending_offset_map[device_id].insert(
                std::make_pair(offset, MemoryBlock(offset, size, 0)));
        }
        this->__async_pending_offset_map[device_id][offset]._pending_events.push_back(event);
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
                //tff::log::Logger::info("device %d reclaim_async memory offset %lld", device_id, block._offset);
                add_memory_heap(device_id, block._offset, block._size, block._free_time);
                iter = this->__async_pending_offset_map[device_id].erase(iter);
            } else {
                ++iter;
            }
        }
    }

    void LLMMemManager::update_used_mem_pool(const FreeBlock &block, const int &device_id,
        const core::memory::MemoryType &type, const FreeBlock &used_block) {
        if (type == core::memory::MemoryType::TFF_MEM_TYPE_KV_CACHE) {
            if (this->_kv_cache_offset[device_id].contains(block._offset)) {
                this->_kv_cache_offset[device_id].erase(this->_kv_cache_offset[device_id].find(block._offset));
            }
            this->_kv_cache_offset[device_id].insert(used_block._offset);
        } else if (type == core::memory::MemoryType::TFF_MEM_TYPE_WORKSPACE){
            if (this->_temp_buffer_offset[device_id].contains(block._offset)) {
                this->_temp_buffer_offset[device_id].erase(this->_temp_buffer_offset[device_id].find(block._offset));
            }
            this->_temp_buffer_offset[device_id].insert(std::make_pair(used_block._offset, used_block._size));
        }else {
            if (this->_const_mem_offset[device_id].contains(block._offset)) {
                this->_const_mem_offset[device_id].erase(this->_const_mem_offset[device_id].find(block._offset));
            }
            this->_const_mem_offset[device_id].insert(used_block._offset);
        }

    }

    void LLMMemManager::add_memory_pool(const int &device_id,
                                        const int64_t &offset, const int64_t &actual_size, const int &end) {
        _mem_buffer_pool[device_id].insert(std::make_pair(offset, MemoryBlock{offset, actual_size, end}));
    }

    void LLMMemManager::add_memory_heap(const int &device_id, const int64_t &offset, const int64_t &actual_size,
        const int &end) {

        this->_memory_heap[device_id].push(MemoryBlock{offset, actual_size, end});
    }

    //
void LLMMemManager::collect(const int &device_id) {
    if (_free_set[device_id].empty() && _memory_heap[device_id].empty()) {
        reclaim_async(device_id);
        if (_memory_heap[device_id].empty()) {
            return;
        }
    }

    std::vector<FreeBlock> pending_blocks;
    while (!_memory_heap[device_id].empty()) {
        auto& block = _memory_heap[device_id].top();
        pending_blocks.push_back({block._offset, block._size});
        _memory_heap[device_id].pop();
    }

    std::sort(pending_blocks.begin(), pending_blocks.end());


    std::vector<FreeBlock> merged_internal;
    for (auto& blk : pending_blocks) {
        if (!merged_internal.empty()) {
            auto& last = merged_internal.back();
            if (last._offset + last._size == blk._offset) {
                last._size += blk._size;
                continue;
            }else if (last._offset + last._size > blk._offset && last._offset < blk._offset){
                tff::log::Logger::error("device %d collect memory error, last offset: %lld, last size: %lld, blk offset: %lld, blk size: %lld",
                    device_id, last._offset, last._size, blk._offset, blk._size);
                continue;
            }
        }
        merged_internal.push_back(blk);
    }


    for (auto& blk : merged_internal) {
        auto it = _free_set[device_id].lower_bound(blk);
        bool merged = false;
        if (it != _free_set[device_id].begin()) {
            const auto prev = std::prev(it);
            if (prev->_offset + prev->_size == blk._offset) {
                FreeBlock new_blk{prev->_offset, prev->_size + blk._size};
                _free_set[device_id].erase(prev);

                if (it != _free_set[device_id].end() && new_blk._offset + new_blk._size == it->_offset) {
                    new_blk._size += it->_size;
                    _free_set[device_id].erase(it);
                }

                _free_set[device_id].insert(new_blk);
                merged = true;
            }
        }

        if (!merged && it != _free_set[device_id].end() && blk._offset + blk._size == it->_offset) {
            FreeBlock new_blk{blk._offset, blk._size + it->_size};
            _free_set[device_id].erase(it);
            _free_set[device_id].insert(new_blk);
            merged = true;
        }

        if (!merged) {
            _free_set[device_id].insert(blk);
        }
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
        std::priority_queue<MemoryBlock, std::vector<MemoryBlock>, std::greater<> > tmp;
        this->_memory_heap[device_id].swap(tmp);
    }
}
