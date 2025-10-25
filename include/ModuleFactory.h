//
// Created by nkk on 2025/4/27.
//

#ifndef DEEP_TFF_MODULEFACTORY_H
#define DEEP_TFF_MODULEFACTORY_H
#include "ExportInc.h"
#include "ModuleObject.h"
#include <any>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

namespace tff::factory {
    class DEEP_TFF_API ModuleFactory : public std::enable_shared_from_this<ModuleFactory>{
    public:
        // 单例访问
        static std::shared_ptr<ModuleFactory> &instance();

    public:
        using Creator = std::function<std::shared_ptr<tff::module::ModuleObject>()>;
        using ModuleMap = std::unordered_map<std::string, Creator>;
        using TypeMap = std::unordered_map<std::string, ModuleMap>;

        template<typename T>
        static void register_module(const std::string &type, const std::string &key) {
            instance()->m_module_func_map[type][key] = []() {
                return std::make_shared<tff::module::ModuleObject>(
                    std::make_shared<T>()
                );
            };
        }

        template<typename T, typename... Args>
        static void register_module(const std::string &type, const std::string &key, Args &&... args) {
            instance()->m_module_func_map[type][key] = [args...]() mutable {
                return std::make_shared<tff::module::ModuleObject>(
                    std::make_shared<T>(args...)
                );
            };
        }

        std::shared_ptr<tff::module::ModuleObject> create_shared(const std::string &_module_type,
                                                                 const std::string &key) {
            auto type_it = m_module_func_map.find(_module_type);
            if (type_it == m_module_func_map.end()) return nullptr;
            auto creator_it = type_it->second.find(key);
            if (creator_it == type_it->second.end()) return nullptr;
            return creator_it->second();
        }

    private:
        ModuleFactory() = default;

    public:
        ModuleFactory(const ModuleFactory &) = delete;

        ModuleFactory(ModuleFactory &&) = delete;

    public:
        // std::unordered_map<std::string, std::function<ptr_type()>> map_;
        TypeMap m_module_func_map;
    };

// #define REGISTER_MODULE_OBJECT_NAME(T) reg_msg_##T##_
// #define REGISTER_MODULE_OBJECT(T, type, key, ...)                              \
//   static tff::factory::ModuleFactory::FactoryRegister_t<void, T> REGISTER_MODULE_OBJECT_NAME(T)(  \
//       type, key, ##__VA_ARGS__);
//
// #define REGISTER_MODULE_OBJECT_BASE(BaseT, T, type, key, ...)                              \
// static tff::factory::ModuleFactory::FactoryRegister_t<BaseT, T> REGISTER_MODULE_OBJECT_NAME(T)(  \
// type, key, ##__VA_ARGS__);


#define REGISTER_MODULE_OBJECT(T, type, key, ...) \
    static struct reg_##T##_##__COUNTER__ { \
            reg_##T##_##__COUNTER__() { \
                ::tff::factory::ModuleFactory::register_module<T>(type, key, ##__VA_ARGS__); \
            } \
    } reg_##T##_##__COUNTER__##_instance;
}
#endif // DEEP_TFF_MODULEFACTORY_H
