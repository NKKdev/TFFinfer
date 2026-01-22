//
// Created by nkk on 2025/9/28.
//

#include "MemBufferAllocatorCUDA.h"
#include "Logger.h"
#include "device/cuda/cudaInc.h"

namespace tff::core::memory {
    void tff::core::memory::MemBufferAllocatorCUDA::release(void *ptr) const {
        if (ptr != nullptr) {
            CudaSafeCall(cudaSetDevice(this->_device_id));
            CudaSafeCall(cudaFree(ptr));
            ptr = nullptr;
        }
    }

    void *tff::core::memory::MemBufferAllocatorCUDA::allocate(const size_t byte_size) const {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        void *ptr = nullptr;
        CudaSafeCall(cudaMalloc(&ptr, byte_size));
        if (ptr == nullptr) {
            size_t free_mem = 0;
            size_t total_mem = 0;
            CudaSafeCall(cudaMemGetInfo(&free_mem, &total_mem));
            tff::log::Logger::error("cudaMalloc() returned NULL,current gpu freemem:%lld, totalmem: %lld", free_mem,
                                    total_mem);
        }
        return ptr;
    }

    void tff::core::memory::MemBufferAllocatorCUDA::memcopy(void *src_ptr, void *dest_ptr, size_t byte_size,
                                                            tff::core::memory::MemCpyKind _memcpy_kind) const {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_HOST2DEVICE) {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        } else if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST) {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        } else {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
        }
    }

    void MemBufferAllocatorCUDA::memcpy_async(const void *src_ptr, void *dest_ptr, size_t byte_size,
                                              tff::core::memory::MemCpyKind _memcpy_kind, void *stream_handle) const {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_HOST2DEVICE) {
            CudaSafeCall(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyHostToDevice, static_cast<
                    cudaStream_t>(stream_handle)));
        } else if (_memcpy_kind == tff::core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST) {
            CudaSafeCall(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, static_cast<
                    cudaStream_t>(stream_handle)));
        } else {
            CudaSafeCall(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, static_cast<
                    cudaStream_t>(stream_handle)));
        }
    }

    void tff::core::memory::MemBufferAllocatorCUDA::memset_zero(void *ptr, size_t byte_size) {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        CudaSafeCall(cudaMemset(ptr, 0, byte_size));
    }
}
