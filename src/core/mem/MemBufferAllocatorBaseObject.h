//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMBUFFERALLOCTORBASEOBJECT_H
#define TFFINFER_MEMBUFFERALLOCTORBASEOBJECT_H
#include "ModuleObject.h"
#include "device/BaseDefine.h"
#include "BaseDefine.h"
namespace tff::core::memory {
    class MemBufferAllocatorBaseObject : public tff::module::ModuleObject {
    public:
        MemBufferAllocatorBaseObject()=default;
        ~MemBufferAllocatorBaseObject() override = default;
    public:
        [[nodiscard]] virtual tff::core::device::DeviceType device_type() const = 0;

        virtual void release(void* ptr) const = 0;

        [[nodiscard]] virtual void *allocate(size_t byte_size) const = 0;

        virtual void memcpy(const void* src_ptr, void* dest_ptr, size_t byte_size,
                            tff::core::memory::MemCpyKind _memcpy_kind) const = 0;

        virtual void memset_zero(void* ptr, size_t byte_size) = 0;
    };
}


#endif //TFFINFER_MEMBUFFERALLOCTORBASEOBJECT_H
