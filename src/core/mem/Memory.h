//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMORY_H
#define TFFINFER_MEMORY_H
#include <cstdint>
#include <memory>
#include <utility>
#include "../device/MemBufferAllocatorBaseObject.h"
namespace tff::core::memory {
    class Memory :public std::enable_shared_from_this<Memory>{
    public:
        explicit Memory() = default;

        explicit Memory(size_t byte_size, void* ptr = nullptr,bool use_external = false,
            std::shared_ptr<device::MemBufferAllocatorBaseObject> allocator = nullptr){
            this->_byte_size = byte_size;
            this->_allocator = std::move(allocator);
            if (use_external) {
                this->_ptr = ptr;
                this->_use_external = use_external;
                this->_is_used = true;
            }
            this->reset();
        }

        virtual ~Memory() {
            if (!_use_external && _allocator != nullptr) {
                _allocator->release(_ptr);
                this->_byte_size = 0;
                _device_type = tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_UNKNOWN;
                _allocator = nullptr;
                //tff::log::Logger::warning("memory buffer destroy");
            }
            //tff::log::Logger::warning("memory destroy");
        }
    public:
        //
        inline void set(size_t byte_size, void* ptr = nullptr,bool use_external = false,
            std::shared_ptr<device::MemBufferAllocatorBaseObject> allocator = nullptr) {
            this->_byte_size = byte_size;
            this->_allocator = std::move(allocator);
            if (use_external) {
                this->_ptr = ptr;
                this->_use_external = use_external;
            }
            this->reset();
        }
        bool allocate();

        void copy_from(const Memory &_mem);

        void *ptr();

        const void *ptr() const;

        inline void set_buffer(void *buffer_ptr) {
            this->_ptr = buffer_ptr;
        }

        size_t byte_size() const;

        std::shared_ptr<device::MemBufferAllocatorBaseObject> allocator() const;

        tff::core::device::DeviceType device_type() const;


        std::shared_ptr<Memory> get_shared_from_this();

        bool is_external() const;
        //
        inline bool is_used() const {
            return this->_is_used;
        }
        //
        inline void reset() {
            this->_is_used = false;
        }
        //
        inline void occupy() {
            this->_is_used = true;
        }

    private:
        size_t _byte_size = 0;
        void* _ptr = nullptr;
        bool _use_external = false;
        tff::core::device::DeviceType _device_type = tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_UNKNOWN;
        std::shared_ptr<device::MemBufferAllocatorBaseObject> _allocator;
        //
        bool _is_used = false;
    };

    inline bool Memory::allocate() {
        if (_allocator && _byte_size > 0) {
            _ptr = _allocator->allocate(_byte_size);
            if (!_ptr) {
                _allocator->memset_zero(_ptr, _byte_size);
            }
        }
        return _ptr != nullptr;
    }

    inline void Memory::copy_from(const Memory &_mem) {
        if (_mem._allocator && _mem._byte_size > 0) {
            this->_allocator = _mem._allocator;
            this->_byte_size = _mem._byte_size;
            if (this->_device_type == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU
                && _mem._device_type == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU) {
                _mem._allocator->memcopy(_mem._ptr, this->_ptr, this->_byte_size, tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE);
                }else if (this->_device_type == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU
                    && _mem._device_type == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU) {
                    _mem._allocator->memcopy(_mem._ptr, this->_ptr, this->_byte_size, tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST);
                }
        }
    }


    inline void * Memory::ptr() {
        return _ptr;
    }

    inline const void * Memory::ptr() const {
        return _ptr;
    }

    inline size_t Memory::byte_size() const {
        return _byte_size;
    }

    inline std::shared_ptr<device::MemBufferAllocatorBaseObject> Memory::allocator() const {
        return _allocator;
    }

    inline tff::core::device::DeviceType Memory::device_type() const {
        return _device_type;
    }

    inline std::shared_ptr<Memory> Memory::get_shared_from_this() {
        return std::static_pointer_cast<Memory>(shared_from_this());
    }

    inline bool Memory::is_external() const {
        return _use_external;
    }
}



#endif //TFFINFER_MEMORY_H