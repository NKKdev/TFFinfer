//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMBUFFERALLOCATORCPU_H
#define TFFINFER_MEMBUFFERALLOCATORCPU_H
#include "../MemBufferAllocatorBaseObject.h"
#include <sys/mman.h>
#include "global/GlobalDefine.h"
namespace tff::core::device {
    static constexpr size_t VMM_INITIAL_RESERVE = 1ULL << 35;
    static constexpr size_t ALIGNMENT = 128;
    class MemBufferAllocatorCPU :public tff::core::device::MemBufferAllocatorBaseObject{
    public:
        MemBufferAllocatorCPU(int device_id):MemBufferAllocatorBaseObject(device_id), _vmm_ptr(nullptr),
                                                _vmm_size(0), _vmm_used(0) {
            _vmm_ptr = mmap(nullptr, VMM_INITIAL_RESERVE,
                         PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                         -1, 0);

            if (_vmm_ptr == MAP_FAILED) {
                constexpr size_t fallback_size = 1ULL << 30;
                _vmm_ptr = mmap(nullptr, fallback_size,
                                     PROT_NONE,
                                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE,
                                     -1, 0);
                if (_vmm_ptr == MAP_FAILED) {
                    tff::log::Logger::fatal("Failed to reserve virtual memory for CPU VMM");
                    throw std::runtime_error("VMM reservation failed");
                }
                this->_vmm_size = fallback_size;
            } else {
                this->_vmm_size = VMM_INITIAL_RESERVE;
            }

            tff::log::Logger::info("CPU VMM reserved %zu bytes at %p", this->_vmm_size, _vmm_ptr);
        };

        ~MemBufferAllocatorCPU() override {
            if (_vmm_ptr && _vmm_ptr != MAP_FAILED) {
                munmap(_vmm_ptr, _vmm_size);
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
        void memcpy_async(const void* src_ptr, void* dest_ptr, size_t byte_size,
                tff::core::memory::MemCpyKind _memcpy_kind, void *stream_handle = nullptr) const override;
        void memset_zero(void *ptr, size_t byte_size) override;
    private:
        size_t _vmm_granularity{};
        void *_vmm_ptr;
        size_t _vmm_size;
        size_t _vmm_used{};
    };

}

#endif //TFFINFER_MEMBUFFERALLOCATORCPU_H