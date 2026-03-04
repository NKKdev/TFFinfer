//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEMBUFFERALLOCTORBASEOBJECT_H
#define TFFINFER_MEMBUFFERALLOCTORBASEOBJECT_H
#include "ModuleObject.h"
#include "device/BaseDefine.h"
#include "../mem/BaseDefine.h"
namespace tff::core::device {
    /**
     * @brief 显式内存分配器基类
     */
    class MemBufferAllocatorBaseObject : public tff::module::ModuleObject {
    public:
        MemBufferAllocatorBaseObject(int device_id): _device_id(device_id){};
        ~MemBufferAllocatorBaseObject() override = default;
    public:
        /**
         * @brief 获取设备类型
         * @return 设备类型
         */
        [[nodiscard]] virtual tff::core::device::DeviceType device_type() const = 0;
        /**
         * @brief 释放内存
         * @param ptr 内存指针
         */
        virtual void release(void* ptr) const = 0;
        /**
         * @brief 分配内存
         * @param byte_size 内存大小
         * @return 内存指针
         */
        [[nodiscard]] virtual void *allocate(size_t byte_size) const = 0;
        /**
         * @brief 分配虚拟内存
         * @param byte_size 内存大小
         * @return 内存指针
         */
        virtual void *allocate_vvm(size_t byte_size) = 0 ;
        /**
         * @brief 内存拷贝
         * @param src_ptr 源内存指针
         * @param dest_ptr 目标内存指针
         * @param byte_size 内存大小
         * @param _memcpy_kind 内存拷贝类型
         */
        virtual void memcopy(void *src_ptr, void *dest_ptr, size_t byte_size,
                            tff::core::memory::MemCpyKind _memcpy_kind = memory::MemCpyKind::TFF_MEM_CPY_TYPE_NORMAL) const = 0;
        /**
         * @brief 异步内存拷贝
         * @param src_ptr 源内存指针
         * @param dest_ptr 目标内存指针
         * @param byte_size 内存大小
         * @param memcpy_kind 内存拷贝类型
         * @param stream_handle 流句柄
         */
        virtual void memcpy_async(const void* src_ptr, void* dest_ptr, size_t byte_size,
        tff::core::memory::MemCpyKind memcpy_kind, void *stream_handle = nullptr) const = 0;
        /**
         * @brief 内存清零
         * @param ptr 内存指针
         * @param byte_size 内存大小
         */
        virtual void memset_zero(void* ptr, size_t byte_size) = 0;
    public:
        int _device_id;
    };
}


#endif //TFFINFER_MEMBUFFERALLOCTORBASEOBJECT_H
