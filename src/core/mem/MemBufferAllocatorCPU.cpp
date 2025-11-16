//
// Created by nkk on 2025/9/28.
//

#include "MemBufferAllocatorCPU.h"

void tff::core::memory::MemBufferAllocatorCPU::release(void *ptr) const {
    if (ptr) {
        free(ptr);ptr = nullptr;
    }
}

void * tff::core::memory::MemBufferAllocatorCPU::allocate(size_t byte_size) const {
    return malloc(byte_size);
}

void tff::core::memory::MemBufferAllocatorCPU::memcpy(void *src_ptr, void *dest_ptr, size_t byte_size,
                                                      tff::core::memory::MemCpyKind _memcpy_kind) const {
    std::memcpy(src_ptr,dest_ptr,byte_size);
}

void tff::core::memory::MemBufferAllocatorCPU::memcpy_async(const void *src_ptr, void *dest_ptr, size_t byte_size,
    tff::core::memory::MemCpyKind _memcpy_kind) const {
}

void tff::core::memory::MemBufferAllocatorCPU::memset_zero(void *ptr, size_t byte_size) {
    std::memset(ptr,0,byte_size);
}
