//
// Created by nkk on 2025/4/27.
//

#ifndef DEEP_TFF_MODULEOBJECT_H
#define DEEP_TFF_MODULEOBJECT_H
#include "ExportInc.h"
#include <string>
#include <memory>
#include <any>
#include <iostream>
namespace tff::module {
    /**
     * 模块对象基类
     */
    class DEEP_TFF_API ModuleObject : public std::enable_shared_from_this<ModuleObject> {
    public:
        ModuleObject() = default;
    public:
        // const std::type_info& get_instance_type() const {
        //     return _impl.type();
        // }
        //
        // //
        // void print_type() const {
        //     std::cout << "ModuleObject holds type: " << _impl.type().name() << std::endl;
        // }
        //
        // // 提取内部对象
        // template<typename T>
        // std::shared_ptr<T> get() const {
        //     try {
        //         return std::any_cast<std::shared_ptr<T>>(_impl);
        //     } catch (const std::bad_any_cast&) {
        //         return nullptr;
        //     }
        // }
        //
        // //
        // template<typename T>
        // explicit ModuleObject(std::shared_ptr<T> impl) : _impl(std::move(impl)) {
        // }

        virtual ~ModuleObject() = default;

        ModuleObject(const ModuleObject &) = delete;

        ModuleObject &operator=(const ModuleObject &) = delete;

    public:
        std::string m_module_name;
    private:
        //std::any _impl;
    };
} // namespace Module
// namespace TFF
#endif // DEEP_TFF_MODULEOBJECT_H
