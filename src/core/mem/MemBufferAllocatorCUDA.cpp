//
// Created by nkk on 2025/9/28.
//

#include "MemBufferAllocatorCUDA.h"
#include "Logger.h"
#include "device/cuda/cudaInc.h"
namespace tff::core::memory {
    void tff::core::memory::MemBufferAllocatorCUDA::release(void *ptr) const {
        CudaSafeCall(cudaFree(ptr));
    }

    void * tff::core::memory::MemBufferAllocatorCUDA::allocate(size_t byte_size) const {
        void *ptr = nullptr;
        CudaSafeCall(cudaMalloc(&ptr, byte_size));
        return ptr;
    }

    void tff::core::memory::MemBufferAllocatorCUDA::memcpy(const void *src_ptr, void *dest_ptr, size_t byte_size,
        tff::core::memory::MemCpyKind _memcpy_kind) const {
        if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_HOST2DEVICE) {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }else if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST) {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }else {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
        }

    }

    void MemBufferAllocatorCUDA::memcpy_async(const void *src_ptr, void *dest_ptr, size_t byte_size,
        tff::core::memory::MemCpyKind _memcpy_kind) const {
        if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_HOST2DEVICE) {
            CudaSafeCall(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        }else if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST) {
            CudaSafeCall(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        }else {
            CudaSafeCall(cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
        }
    }

    void tff::core::memory::MemBufferAllocatorCUDA::memset_zero(void *ptr, size_t byte_size) {
        CudaSafeCall(cudaMemset(ptr, 0, byte_size));
    }
}

