//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_DEVICE_BASEDEFINE_H
#define TFFINFER_DEVICE_BASEDEFINE_H
#include <stddef.h>
#include <string>
#include "ModuleFactory.h"
namespace tff::core::device {

    enum DeviceSchedulingPriority {
        TFF_DEVICE_PRIORITY_DEFAULT = 0,
        TFF_DEVICE_PRIORITY_CPU = 1,
        TFF_DEVICE_PRIORITY_GPU = 2,
    };
    //
    enum DeviceType {
        TFF_BACKEND_DEVICE_TYPE_UNKNOWN = 0,
        TFF_BACKEND_DEVICE_TYPE_CPU = 1,
        TFF_BACKEND_DEVICE_TYPE_GPU = 2,
    };

    //
    struct DeviceCaps {
        // asynchronous operations
        bool async;
        // pinned host buffer
        bool host_buffer;
        // creating buffers from host ptr
        bool buffer_from_host_ptr;
        // event synchronization
        bool events;
    };

    //
    struct DeviceProperties {
        // device name
        const char *name;
        // device description
        const char *description;
        // device free memory in bytes
        size_t memory_free;
        // device total memory in bytes
        size_t memory_total;
        // device type
        enum DeviceType type;
        // device id
        const char *device_id;
        // device capabilities
        struct DeviceCaps caps;
    };



}

#endif //TFFINFER_DEVICE_BASEDEFINE_H
