//
// Created by nkk on 2025/9/28.
//

#include "MemBufferAllocatorCUDA.h"
#include "Logger.h"
#include "device/cuda/cudaInc.h"

namespace tff::core::device {
    static bool is_valid_device_pointer(void *ptr) {
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
        if (err == cudaSuccess) {
            return attributes.type == cudaMemoryTypeDevice;
        }
        return false;
    }

    static bool is_valid_host_pointer(void *ptr) {
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);
        if (err == cudaSuccess) {
            return attributes.type == cudaMemoryTypeHost || attributes.type == cudaMemoryTypeUnregistered;
        }
        return true;
    }

    //
    void MemBufferAllocatorCUDA::release(void *ptr) const {
        if (ptr != nullptr) {
            CudaSafeCall(cudaSetDevice(this->_device_id));
            CudaSafeCall(cudaFree(ptr));
            ptr = nullptr;
        }
    }

    void *MemBufferAllocatorCUDA::allocate(const size_t byte_size) const {
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

    //
    void *MemBufferAllocatorCUDA::allocate_vvm(size_t byte_size) {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        byte_size                   = ALIGNMENT * ((byte_size + ALIGNMENT - 1) / ALIGNMENT);
        if ((this->_vmm_used + byte_size) > this->_vmm_size) {
            size_t reserve_size = this->_vmm_granularity * ((byte_size + this->_vmm_granularity - 1) / this->_vmm_granularity);
            CUmemAllocationProp prop = {};
            prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id         = this->_device_id;
            CUmemGenericAllocationHandle handle;
            auto error = cuMemCreate(&handle, reserve_size, &prop, 0);
            if (error != CUDA_SUCCESS) {
                tff::log::Logger::error("cuMemCreate() returned error code %d", error);
                return nullptr;
            }
            CUdeviceptr start_ptr = (CUdeviceptr) ((char *) (this->_vmm_ptr) + this->_vmm_size);
            error = cuMemMap(start_ptr, reserve_size, 0, handle, 0);
            if (error != CUDA_SUCCESS) {
                tff::log::Logger::error("cuMemMap() returned error code %d", error);
                return nullptr;
            }
            cuMemRelease(handle);
            CUmemAccessDesc access = {};
            access.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id     = this->_device_id;
            access.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            error = cuMemSetAccess((CUdeviceptr) ((char *) (this->_vmm_ptr) + this->_vmm_size), reserve_size, &access, 1);
            if (error != CUDA_SUCCESS) {
                tff::log::Logger::error("cuMemSetAccess() returned error code %d", error);
                return nullptr;
            }
            this->_vmm_size += reserve_size;
        }
        void * ptr   = (void *) ((CUdeviceptr) ((char *) (this->_vmm_ptr) + this->_vmm_used));
        CudaSafeCall(cudaMemset(ptr, 0, byte_size));
        this->_vmm_used += byte_size;
        return ptr;

    }

    void MemBufferAllocatorCUDA::memcopy(void *src_ptr, void *dest_ptr, size_t byte_size,
                                                            memory::MemCpyKind _memcpy_kind) const {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        if (_memcpy_kind == memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE) {
            if (!is_valid_device_pointer(dest_ptr)) {
                tff::log::Logger::error("memory pointer type is invalid!!");
                return;
            }
            if (!is_valid_host_pointer(src_ptr)) {
                tff::log::Logger::error("memory pointer type is invalid!!");
                return;
            }
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
        } else if (_memcpy_kind == memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST) {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        } else {
            CudaSafeCall(cudaMemcpy(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice));
        }
    }

    void MemBufferAllocatorCUDA::memcpy_async(const void *src_ptr, void *dest_ptr, size_t byte_size,
                                              memory::MemCpyKind memcpy_kind, void *stream_handle) const {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        if (memcpy_kind == memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE) {
            CudaSafeCall(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyHostToDevice, static_cast<
                    cudaStream_t>(stream_handle)));
        } else if (memcpy_kind == memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST) {
            CudaSafeCall(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, static_cast<
                    cudaStream_t>(stream_handle)));
        } else {
            CudaSafeCall(
                cudaMemcpyAsync(dest_ptr, src_ptr, byte_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, static_cast<
                    cudaStream_t>(stream_handle)));
        }
    }

    void MemBufferAllocatorCUDA::memset_zero(void *ptr, size_t byte_size) {
        CudaSafeCall(cudaSetDevice(this->_device_id));
        CudaSafeCall(cudaMemset(ptr, 0, byte_size));
    }
}
