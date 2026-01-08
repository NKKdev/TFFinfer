//
// Created by nkk on 2025/11/16.
//

#include "LLMMemManager.h"
#include "global/GlobalDefine.h"
#include "device/DeviceBaseObject.h"
namespace tff::core::runtime {
    //REGISTER_MODULE_OBJECT(LLMMemManager, tff::module::ModuleObject, WEIGHT_MEM_BUFFER_MANAGER_FLAG,WEIGHT_MEM_BUFFER_MANAGER_FLAG);
    bool LLMMemManager::init(const int &device_id) {
        auto mem_allocator = this->_devices_map[device_id]->get_device_buffer_allocator(device_id);
        this->_mem_buffer_map[device_id] = mem_allocator->allocate(this->_current_offset[device_id]);
        if (this->_mem_buffer_map[device_id] == nullptr) {
            tff::log::Logger::error("device : %d init mem buffer failed!!", device_id);
            return false;
        }
        _is_initialized = true;
        return true;
    }

    int LLMMemManager::allocate_memory(size_t size, int start, int end, const int &device_id) {
        std::lock_guard<std::mutex> lock(_mutex);

        while (!_free_heap[device_id].empty() && _free_heap[device_id].top()._free_time < start) {
            auto block = _free_heap[device_id].top();
            _free_heap[device_id].pop();
            _free_set[device_id].insert({block._offset, block._size});
        }

        size_t actual_size = size;
        if (this->_devices_map[device_id]->get_device_type(device_id) != device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU) {
            actual_size = align_up(size, _alignment);
        }

        size_t offset;
        auto& free_set = _free_set[device_id];
        auto it = free_set.lower_bound({0, actual_size});

        if (it != free_set.end()) {
            size_t candidate_offset = it->offset;
            size_t candidate_size = it->size;

            if (candidate_size >= actual_size + _alignment) {
                offset = candidate_offset;
                size_t remaining_offset = candidate_offset + actual_size;
                size_t remaining_size = candidate_size - actual_size;

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

        _free_heap[device_id].push({offset, actual_size, end});

        return static_cast<int>(offset);
    }
}
