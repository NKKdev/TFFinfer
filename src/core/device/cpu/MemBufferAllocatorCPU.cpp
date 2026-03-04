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
        byte_size = ALIGNMENT * ((byte_size + ALIGNMENT - 1) / ALIGNMENT);
        if (_vmm_used + byte_size > this->_vmm_size) {
            tff::log::Logger::error(
                "CPU VMM exhausted: requested %zu, used %zu, reserved %zu",
                byte_size, _vmm_used, _vmm_size
            );
            return nullptr;
        }

        char* alloc_ptr = static_cast<char*>(this->_vmm_ptr) + _vmm_used;

        if (mprotect(alloc_ptr, byte_size, PROT_READ | PROT_WRITE) != 0) {
            tff::log::Logger::error("mprotect failed to commit VMM pages");
            return nullptr;
        }

        std::memset(alloc_ptr, 0, byte_size);
        _vmm_used += byte_size;
        return alloc_ptr;
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