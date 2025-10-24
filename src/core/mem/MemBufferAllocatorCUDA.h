//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMBUFFERALLOCATORGPU_H
#define TFFINFER_MEMBUFFERALLOCATORGPU_H
#include "MemBufferAllocatorBaseObject.h"
#include "ModuleFactory.h"
namespace tff::core::memory {
    class MemBufferAllocatorCUDA : public tff::core::memory::MemBufferAllocatorBaseObject {
    public:
        MemBufferAllocatorCUDA() = default;

        ~MemBufferAllocatorCUDA() override = default;

    public:
        [[nodiscard]] inline tff::core::device::DeviceType device_type() const override {
            return tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU;
        }

        void release(void *ptr) const override;

        [[nodiscard]] void *allocate(size_t byte_size) const override;

        void memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
                    tff::core::memory::MemCpyKind _memcpy_kind) const override;

        void memset_zero(void *ptr, size_t byte_size) override;
    };
    REGISTER_MODULE_OBJECT(MemBufferAllocatorCUDA, "MEMORY", "CUDA")
}


#endif //TFFINFER_MEMBUFFERALLOCATORGPU_H
