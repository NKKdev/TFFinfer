//
// Created by nkk on 2025/4/27.
//

#ifndef DEEP_TFF_MODULEOBJECT_H
#define DEEP_TFF_MODULEOBJECT_H
#include "ExportInc.h"
#include <string>
#include <memory>
#include <any>

namespace tff::module {
    class DEEP_TFF_API ModuleObject : public std::enable_shared_from_this<ModuleObject> {
    public:
        ModuleObject() = default;

        //
        template<typename T>
        explicit ModuleObject(std::shared_ptr<T> impl) : _impl(std::move(impl)) {
        }

        virtual ~ModuleObject() = default;

        ModuleObject(const ModuleObject &) = delete;

        ModuleObject &operator=(const ModuleObject &) = delete;

    public:
        std::string m_module_name;

    private:
        std::any _impl;
    };
} // namespace Module
// namespace TFF
#endif // DEEP_TFF_MODULEOBJECT_H
