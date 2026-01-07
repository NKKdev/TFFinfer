//
// Created by nkk on 2025/11/16.
//

#include "LLMMemManager.h"
#include "global/GlobalDefine.h"
#include "device/DeviceBaseObject.h"
namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMMemManager, tff::module::ModuleObject, WEIGHT_MEM_BUFFER_MANAGER_FLAG,WEIGHT_MEM_BUFFER_MANAGER_FLAG);

    bool LLMMemManager::init(const int &device_id) {
        auto mem_allocator = this->_devices_map[device_id]->get_device_buffer_allocator(device_id);
        this->_mem_buffer_map[device_id] = mem_allocator->allocate(this->_current_offset[device_id]);
        if (this->_mem_buffer_map[device_id] == nullptr) {
            tff::log::Logger::error("device : %d init mem buffer failed!!", device_id);
            return false;
        }
        return true;
    }

    int LLMMemManager::allocate_memory(size_t size, int start, int end, const int &device_id) {
        std::lock_guard<std::mutex> lock(_mutex);

        while (!_free_heap[device_id].empty() && _free_heap[device_id].top()._free_time <= start) {
            auto block = _free_heap[device_id].top();
            _free_heap[device_id].pop();
            _free_list[device_id].push_back(block._offset);
        }

        size_t offset;
        if (!_free_list.empty()) {
            offset = _free_list[device_id].back();
            _free_list[device_id].pop_back();
        } else {
            offset = _current_offset[device_id];
            _current_offset[device_id] += align_up(size, _alignment);
        }

        _free_heap[device_id].push({offset, end});

        return offset;
    }
}
