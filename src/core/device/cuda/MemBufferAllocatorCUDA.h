//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMBUFFERALLOCATORGPU_H
#define TFFINFER_MEMBUFFERALLOCATORGPU_H
#include "../MemBufferAllocatorBaseObject.h"
#include "ModuleFactory.h"
#include "global/GlobalDefine.h"

namespace tff::core::device {
    static constexpr size_t VMM_INITIAL_RESERVE = 1ULL << 35;
    static constexpr size_t ALIGNMENT = 128;
    class MemBufferAllocatorCUDA : public tff::core::device::MemBufferAllocatorBaseObject {
    public:
        MemBufferAllocatorCUDA(int device_id) : MemBufferAllocatorBaseObject(device_id), _vmm_ptr(0),
                                                _vmm_size(0), _vmm_used(0) {
            int device_vmm = 0;
            CUdevice device;
            cuDeviceGet(&device, device_id);
            cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device);

            if (device_vmm) {
                CUmemAllocationProp alloc_prop = {};
                alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
                alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                alloc_prop.location.id = device_id;
                cuMemGetAllocationGranularity(&this->_vmm_granularity, &alloc_prop,
                                              CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
                auto error = cuMemAddressReserve(&this->_vmm_ptr, VMM_INITIAL_RESERVE, 128, 0, 0);
                if (error != CUDA_SUCCESS) {
                    tff::log::Logger::warning("current device vmm addr reserve failed!");
                }
            } else {
                tff::log::Logger::warning("current device don't support vmm!");
            }
        };

        ~MemBufferAllocatorCUDA() {
            if (this->_vmm_ptr != 0) {
                cuMemUnmap(this->_vmm_ptr, this->_vmm_size);
                cuMemAddressFree(this->_vmm_ptr, VMM_INITIAL_RESERVE);
            }
        }

    public:
        [[nodiscard]] inline tff::core::device::DeviceType device_type() const override {
            return tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU;
        }

        void release(void *ptr) const override;

        [[nodiscard]] void *allocate(size_t byte_size) const override;

        void *allocate_vvm(size_t byte_size) override;

        void memcopy(void *src_ptr, void *dest_ptr, size_t byte_size,
                     tff::core::memory::MemCpyKind _memcpy_kind = memory::MemCpyKind::TFF_MEM_CPY_TYPE_NORMAL) const override;

        //
        void memcpy_async(const void *src_ptr, void *dest_ptr, size_t byte_size,
                          tff::core::memory::MemCpyKind _memcpy_kind, void *stream_handle = nullptr) const override;

        void memset_zero(void *ptr, size_t byte_size) override;

    private:
        size_t _vmm_granularity{};
        CUdeviceptr _vmm_ptr;
        size_t _vmm_size;
        size_t _vmm_used;
    };
}


#endif //TFFINFER_MEMBUFFERALLOCATORGPU_H
