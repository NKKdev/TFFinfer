//
// Created by nkk on 2025/9/28.
//

#include "MemBufferAllocatorCPU.h"

void tff::core::memory::MemBufferAllocatorCPU::release(void *ptr) const {
}

void * tff::core::memory::MemBufferAllocatorCPU::allocate(size_t byte_size) const {
}

void tff::core::memory::MemBufferAllocatorCPU::memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
    tff::core::memory::MemCpyKind _memcpy_kind) const {
}

void tff::core::memory::MemBufferAllocatorCPU::memset_zero(void *ptr, size_t byte_size) {
}
