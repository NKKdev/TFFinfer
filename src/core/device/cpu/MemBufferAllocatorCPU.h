//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMBUFFERALLOCATORCPU_H
#define TFFINFER_MEMBUFFERALLOCATORCPU_H
#include "../MemBufferAllocatorBaseObject.h"
#include "ModuleFactory.h"
#include "global/GlobalDefine.h"
namespace tff::core::device {
    class MemBufferAllocatorCPU :public tff::core::device::MemBufferAllocatorBaseObject{
    public:
        MemBufferAllocatorCPU(int device_id):MemBufferAllocatorBaseObject(device_id){};

        ~MemBufferAllocatorCPU() override = default;

    public:
        [[nodiscard]] inline tff::core::device::DeviceType device_type() const override {
            return tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU;
        }

        void release(void *ptr) const override;

        [[nodiscard]] void *allocate(size_t byte_size) const override;

        void *allocate_vvm(size_t byte_size) override;

        void memcopy(void *src_ptr, void *dest_ptr, size_t byte_size,
                    tff::core::memory::MemCpyKind _memcpy_kind = memory::MemCpyKind::TFF_MEM_CPY_TYPE_NORMAL) const override;
        void memcpy_async(const void* src_ptr, void* dest_ptr, size_t byte_size,
                tff::core::memory::MemCpyKind _memcpy_kind, void *stream_handle = nullptr) const override;
        void memset_zero(void *ptr, size_t byte_size) override;
    };

}

#endif //TFFINFER_MEMBUFFERALLOCATORCPU_H