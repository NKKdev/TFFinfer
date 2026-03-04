//
// Created by nkk on 2025/10/28.
//

#include "DeviceCPU.h"
#include "FunctionFactory.h"
#include "global/ModelGlobalVar.h"
#include "MemBufferAllocatorCPU.h"
#include <iostream>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#elif __APPLE__
#include <sys/sysctl.h>
#include <mach/mach.h>
#elif __linux__
#include <fstream>
#include <string>
#include <unistd.h>
#else
#error "Unsupported platform"
#endif

namespace tff::core::device::cpu {
    REGISTER_MODULE_OBJECT(DeviceCPU, DeviceBaseObject, DEVICE_BACKEND_FLAG, DEVICE_BACKEND_TYPE_CPU);

    struct MemoryInfo {
        uint64_t total; // bytes
        uint64_t available; // bytes
        uint64_t used; // bytes
    };
    bool starts_with(const std::string& line, const std::string& prefix) {
        if (line.size() < prefix.size()) return false;
        size_t start = 0;
        while (start < line.size() && std::isspace(static_cast<unsigned char>(line[start]))) {
            ++start;
        }
        if (line.size() - start < prefix.size()) return false;
        return line.compare(start, prefix.size(), prefix) == 0;
    }
    static MemoryInfo getMemoryInfo() {
        MemoryInfo mem = {0, 0, 0};

#ifdef _WIN32
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        GlobalMemoryStatusEx(&statex);

        mem.total = statex.ullTotalPhys;
        mem.available = statex.ullAvailPhys;
        mem.used = mem.total - mem.available;

#elif __linux__
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        uint64_t mem_total = 0, mem_available = 0;

        while (std::getline(meminfo, line)) {
            //tff::log::Logger::info("cpu device info: %s", line.c_str());
            if (starts_with(line, "MemTotal:")) {
                size_t pos = line.find_first_of(":");
                size_t end_pos = line.find_last_of("kb");
                if (pos != std::string::npos) {
                    std::string digital = line.substr(pos + 1, end_pos - pos - 1);
                    mem_total = std::stoull(digital) * 1024;
                }
            } else if (starts_with(line, "MemAvailable:")) {
                size_t pos = line.find_first_of(":");
                size_t end_pos = line.find_last_of("kb");
                if (pos != std::string::npos) {
                    std::string digital = line.substr(pos + 1, end_pos - pos - 1);
                    mem_available = std::stoull(digital) * 1024;
                }
            }
        }

        mem.total = mem_total;
        mem.available = mem_available;
        mem.used = (mem_total > mem_available) ? (mem_total - mem_available) : 0;
#endif

        return mem;
    }

    static double toGB(uint64_t bytes) {
        return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    }

    void DeviceCPU::get_device_id(std::vector<int> &_device_list) {
        _device_list.push_back(-1);
    }

    const char *DeviceCPU::get_device_name(size_t _device_id) {
        return "CPU";
    }

    const char *DeviceCPU::get_device_description(size_t _device_id) {
    }

    void DeviceCPU::get_device_mem(size_t _device_id, size_t *_free_mem, size_t *_total_mem) {
        MemoryInfo mem = getMemoryInfo();
        *_total_mem = (mem.total);
        *_free_mem = (mem.available);
    }

    tff::core::device::DeviceType DeviceCPU::get_device_type(size_t _device_id) {
        return device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU;
    }

    DeviceType DeviceCPU::device_type() {
        std::vector<int> device_list;
        get_device_id(device_list);
        if (device_list.empty()) {
            return tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_UNKNOWN;
        }
        return get_device_type(device_list[0]);
    }

    std::string DeviceCPU::get_device_type_flag(size_t _device_id) {
        return DEVICE_BACKEND_TYPE_CPU;
    }

    void DeviceCPU::get_device_props(size_t _device_id, tff::core::device::DeviceProperties &_device_props) {
    }

    void DeviceCPU::device_init() {
        std::vector<int> device_list;
        get_device_id(device_list);
        for (size_t i = 0; i < device_list.size(); i++) {
            auto mem_buffer_allocator = std::make_shared<MemBufferAllocatorCPU>(device_list[i]);
            this->_mem_buffer_allocators.insert(std::make_pair(device_list[i], mem_buffer_allocator));
        }
    }

    std::shared_ptr<MemBufferAllocatorBaseObject> DeviceCPU::get_device_buffer_allocator(
        const int &device_id) {
        return this->_mem_buffer_allocators[device_id];
    }

    std::shared_ptr<DeviceStream> DeviceCPU::create_stream(int device_id) {
        return std::make_shared<CPUStream>(device_id);
    }

    std::shared_ptr<DeviceEvent> DeviceCPU::create_event(int device_id) {
        return std::make_shared<CPUEvent>(device_id);
    }
    float DeviceCPU::elapsed_time(const std::shared_ptr<DeviceEvent> &start, const std::shared_ptr<DeviceEvent> &stop) {
        return 0.0f;
    }
}
