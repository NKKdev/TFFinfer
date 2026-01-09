//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMORY_H
#define TFFINFER_MEMORY_H
#include <cstdint>
#include <memory>
#include <utility>
#include "MemBufferAllocatorBaseObject.h"
namespace tff::core::memory {
    class Memory :public std::enable_shared_from_this<Memory>{
    public:
        explicit Memory() = default;

        explicit Memory(size_t byte_size, void* ptr = nullptr,bool use_external = false,
            std::shared_ptr<MemBufferAllocatorBaseObject> allocator = nullptr){
            this->byte_size_ = byte_size;
            this->_allocator = std::move(allocator);
            if (use_external) {
                this->ptr_ = ptr;
                this->use_external_ = use_external;
                this->_is_used = true;
            }
            this->reset();
        }

        virtual ~Memory() {
            if (!use_external_ && _allocator != nullptr) {
                _allocator->release(ptr_);
                this->byte_size_ = 0;
                device_type_ = tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_UNKNOWN;
                _allocator = nullptr;
                //tff::log::Logger::warning("memory buffer destroy");
            }
            //tff::log::Logger::warning("memory destroy");
        }
    public:
        //
        inline void set(size_t byte_size, void* ptr = nullptr,bool use_external = false,
            std::shared_ptr<MemBufferAllocatorBaseObject> allocator = nullptr) {
            this->byte_size_ = byte_size;
            this->_allocator = std::move(allocator);
            if (use_external) {
                this->ptr_ = ptr;
                this->use_external_ = use_external;
            }
            this->reset();
        }
        bool allocate();

        void copy_from(const Memory &_mem);

        void *ptr();

        const void *ptr() const;

        size_t byte_size() const;

        std::shared_ptr<MemBufferAllocatorBaseObject> allocator() const;

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
        size_t byte_size_ = 0;
        void* ptr_ = nullptr;
        bool use_external_ = false;
        tff::core::device::DeviceType device_type_ = tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_UNKNOWN;
        std::shared_ptr<MemBufferAllocatorBaseObject> _allocator;
        //
        bool _is_used = false;
    };

    inline bool Memory::allocate() {
        if (_allocator && byte_size_ > 0) {
            ptr_ = _allocator->allocate(byte_size_);
            if (!ptr_) {
                _allocator->memset_zero(ptr_, byte_size_);
            }
        }
        return ptr_ != nullptr;
    }

    inline void Memory::copy_from(const Memory &_mem) {
        if (_mem._allocator && _mem.byte_size_ > 0) {
            this->_allocator = _mem._allocator;
            this->byte_size_ = _mem.byte_size_;
            if (this->device_type_ == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU
                && _mem.device_type_ == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU) {
                _mem._allocator->memcopy(_mem.ptr_, this->ptr_, this->byte_size_, tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_HOST2DEVICE);
                }else if (this->device_type_ == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU
                    && _mem.device_type_ == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU) {
                    _mem._allocator->memcopy(_mem.ptr_, this->ptr_, this->byte_size_, tff::core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2HOST);
                }
        }
    }


    inline void * Memory::ptr() {
        return ptr_;
    }

    inline const void * Memory::ptr() const {
        return ptr_;
    }

    inline size_t Memory::byte_size() const {
        return byte_size_;
    }

    inline std::shared_ptr<MemBufferAllocatorBaseObject> Memory::allocator() const {
        return _allocator;
    }

    inline tff::core::device::DeviceType Memory::device_type() const {
        return device_type_;
    }

    inline std::shared_ptr<Memory> Memory::get_shared_from_this() {
        return std::static_pointer_cast<Memory>(shared_from_this());
    }

    inline bool Memory::is_external() const {
        return use_external_;
    }
}



#endif //TFFINFER_MEMORY_H