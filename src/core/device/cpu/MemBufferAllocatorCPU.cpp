//
// Created by nkk on 2025/9/28.
//

#include "MemBufferAllocatorCPU.h"
namespace tff::core::device {
    void MemBufferAllocatorCPU::release(void *ptr) const {
        if (ptr) {
            free(ptr);ptr = nullptr;
        }
    }

    void * MemBufferAllocatorCPU::allocate(size_t byte_size) const {
        return malloc(byte_size);
    }

    void *MemBufferAllocatorCPU::allocate_vvm(size_t byte_size) {
        return malloc(byte_size);
    }

    void MemBufferAllocatorCPU::memcopy(void *src_ptr, void *dest_ptr, size_t byte_size,
                                                          core::memory::MemCpyKind _memcpy_kind) const {
        try {
            memcpy(dest_ptr, src_ptr,byte_size);
        }
        catch (std::exception& e) {
            tff::log::Logger::error("memcpy error: %s",e.what());
        }

    }

    void MemBufferAllocatorCPU::memcpy_async(const void *src_ptr, void *dest_ptr, size_t byte_size,
        core::memory::MemCpyKind _memcpy_kind, void *stream_handle) const {
    }

    void MemBufferAllocatorCPU::memset_zero(void *ptr, size_t byte_size) {
        std::memset(ptr,0,byte_size);
    }
}