//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_BUFFER_H
#define TFFINFER_BUFFER_H
#include "BaseDefine.h"
#include "../../core/device/MemBufferAllocatorBaseObject.h"
#include "log/Logger.h"
#include "device/cuda/cudaInc.h"
namespace tff::kernel {
    template<typename T>
    class Buffer {
    public:
        explicit Buffer(T *buffer,
                        const std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> &allocator =
                                nullptr) :
                                           _access(base::BufferAccess::kNotOwned),
                                           _allocator(allocator) {
            _buffer = buffer;
        }

        // Copies from device to host: reading the device buffer a-synchronously
        void ReadAsync(const size_t size, T *host, const size_t offset = 0) const {
            if (_access == base::BufferAccess::kWriteOnly) {
                tff::log::Logger::error("Buffer: reading from a write-only buffer");
            }
            this->_allocator->memcpy_async(*_buffer, host, size * sizeof(T), tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST);
        }

        // Copies from device to host: reading the device buffer
        void Read(const size_t size, T *host, const size_t offset = 0) const {
            if (_access == base::BufferAccess::kWriteOnly) {
                tff::log::Logger::error("Buffer: reading from a write-only buffer");
            }
            this->_allocator->memcopy(*_buffer, host, size * sizeof(T), tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST);
        }

        void WriteAsync(const size_t size, const T *host, const size_t offset = 0) {
            if (_access == base::BufferAccess::kReadOnly) {
                tff::log::Logger::error("Buffer: writing to a read-only buffer");
            }

            this->_allocator->memcpy_async(host, *_buffer, size * sizeof(T), tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE);
        }

        void Write(const size_t size, const T *host, const size_t offset = 0) {
            if (_access == base::BufferAccess::kReadOnly) {
                tff::log::Logger::error("Buffer: writing to a read-only buffer");
            }

            this->_allocator->memcopy(host, *_buffer, size * sizeof(T), tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE);
        }

        // Copies the contents of this buffer into another device buffer
        void CopyToAsync(const size_t size, const Buffer<T> &destination) const {
            this->_allocator->memcpy_async(*_buffer, destination(), size * sizeof(T), tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2DEVICE);
        }

        void CopyTo(const size_t size, const Buffer<T> &destination) const {
            this->_allocator->memcopy(*_buffer, destination(), size * sizeof(T), tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2DEVICE);
        }

        // Accessor to the private data-member
        //const T &operator()() const { return *_buffer; }

    private:
        T *_buffer;
        base::BufferAccess _access;
        std::shared_ptr<tff::core::device::MemBufferAllocatorBaseObject> _allocator;
    };
}

#endif //TFFINFER_BUFFER_H
