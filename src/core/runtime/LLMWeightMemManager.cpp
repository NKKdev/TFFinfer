//
// Created by nkk on 2025/11/16.
//

#include "LLMWeightMemManager.h"
#include "global/GlobalDefine.h"
#include "device/DeviceBaseObject.h"
namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(LLMWeightMemManager, tff::module::ModuleObject, WEIGHT_MEM_BUFFER_MANAGER_FLAG,WEIGHT_MEM_BUFFER_MANAGER_FLAG);
    bool LLMWeightMemManager::init(const size_t &buffer_size) {
        std::lock_guard<std::mutex> lock(_mutex);
        bool ret = true;
        if (buffer_size == 0) {
            ret = false;
        }
        ret &= init_cpu_buffer(buffer_size);
        ret &= init_gpu_buffer(buffer_size);
        return ret;
    }
    bool LLMWeightMemManager::init_cpu_buffer(const size_t &buffer_size) {
        bool ret = true;
        auto cpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
        if (cpu_device) {
            this->_cpu_mapped_buffer.reserve(MAX_PREFETCH_BUFFER_SIZE);
            this->_pinned_buffer.reserve(MAX_PREFETCH_BUFFER_SIZE);
            auto allocator = cpu_device->get_device_buffer_allocator();
            for (size_t i = 0; i < MAX_PREFETCH_BUFFER_SIZE; i++) {
                //auto &prefetch_buffer = this->_cpu_mapped_buffer[i];
                auto prefetch_buffer = std::make_shared<tff::core::memory::Memory>(buffer_size, nullptr, false, allocator);
                if (prefetch_buffer == nullptr) {
                    continue;
                }
                ret &= prefetch_buffer->allocate();
                if (ret == true) {
                    this->_cpu_mapped_buffer.push_back(prefetch_buffer);
                }

            }
            return ret;
        }else {
            ret = false;
        }
        return ret;
    }

    bool LLMWeightMemManager::init_gpu_buffer(const size_t &buffer_size) {
        bool ret = true;
        auto gpu_device = tff::factory::ModuleFactory::instance()->create_shared<tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG,
            tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
        if (gpu_device) {
            this->_gpu_buffer.reserve(MAX_PREFETCH_BUFFER_SIZE);
            auto allocator = gpu_device->get_device_buffer_allocator();
            for (size_t i = 0; i < MAX_PREFETCH_BUFFER_SIZE; i++) {
                //auto &prefetch_buffer = this->_gpu_buffer[i];
                auto prefetch_buffer = std::make_shared<tff::core::memory::Memory>(buffer_size, nullptr, false, allocator);
                if (prefetch_buffer == nullptr) {
                    continue;
                }
                ret &= prefetch_buffer->allocate();
                if (ret == true) {
                    this->_gpu_buffer.push_back(prefetch_buffer);
                }

            }
            return ret;
        }else {
            ret = false;
        }
        return ret;

    }
}
