//
// Created by nkk on 2026/2/19.
//

#ifndef TFFINFER_DEVICEMANAGER_H
#define TFFINFER_DEVICEMANAGER_H
#include <set>
#include <vector>

#include "BaseDefine.h"
#include "DeviceBaseObject.h"
#include "ModuleObject.h"
#include "global/GlobalDefine.h"

namespace tff::core::device {
    /**
     * 设备管理器
     */
    class DeviceManager : public tff::module::ModuleObject {
    public:
        DeviceManager() {
            auto gpu_device = tff::factory::ModuleFactory::instance()
                    ->create_shared<tff::core::device::DeviceBaseObject>(
                        DEVICE_BACKEND_FLAG,
                        tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
            if (gpu_device) {
                gpu_device->device_init();
                this->_devices.insert(gpu_device);
                std::vector<int> device_ids;
                gpu_device->get_device_id(device_ids);
                for (auto device_id: device_ids) {
                    this->_device_map.insert(std::make_pair(device_id, gpu_device));
                }
            }

            auto cpu_device = tff::factory::ModuleFactory::instance()
                    ->create_shared<tff::core::device::DeviceBaseObject>(
                        DEVICE_BACKEND_FLAG,
                        tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
            if (cpu_device) {
                cpu_device->device_init();
                this->_devices.insert(cpu_device);
                std::vector<int> device_ids;
                cpu_device->get_device_id(device_ids);
                for (auto device_id: device_ids) {
                    this->_device_map.insert(std::make_pair(device_id, cpu_device));
                }
            }
        }

        ~DeviceManager() = default;

    public:
        /**
         * 获取设备
         * @param device_id
         * @return
         */
        inline std::shared_ptr<tff::core::device::DeviceBaseObject>
        &get_device(const int &device_id) {
            return _device_map[device_id];
        }

        /**
         * 获取设备列表
         * @return
         */
        inline std::unordered_map<int, std::shared_ptr<DeviceBaseObject> > &devices() {
            return this->_device_map;
        }

        /**
         * 获取默认设备**
         * 获取默认设备
         * @return
         */
        inline int prefered_device() {
            auto device = *_devices.begin();
            std::vector<int> device_ids;
            device->get_device_id(device_ids);
            return device_ids[0];
        }

    protected:
        std::vector<std::shared_ptr<tff::core::device::DeviceBaseObject> > _device_list;
        std::unordered_map<int,
            std::shared_ptr<tff::core::device::DeviceBaseObject> > _device_map;

        std::set<std::shared_ptr<tff::core::device::DeviceBaseObject>, tff::core::device::DeviceComparator> _devices;
    };
}
#endif //TFFINFER_DEVICEMANAGER_H
