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
    class DEEP_TFF_API ModuleFactory {
    public:
        // 单例
        static std::shared_ptr<ModuleFactory> &instance();

        // 删除拷贝和移动
        ModuleFactory(const ModuleFactory &) = delete;

        ModuleFactory &operator=(const ModuleFactory &) = delete;

        ModuleFactory(ModuleFactory &&) = delete;

        ModuleFactory &operator=(ModuleFactory &&) = delete;

    private:
        ModuleFactory() = default;

    public:

        template<typename T, typename Base, typename... Args>
        static void register_type(const std::string &type, const std::string &key, Args &&... args) {
            auto &creators = instance()->get_or_create_creator_list<Base>(type);
            creators[key] = [args...]() -> std::shared_ptr<Base> {
                return std::make_shared<T>(args...);
            };
        }

        // 无参版本
        template<typename T, typename Base>
        static void register_type(const std::string &type, const std::string &key) {
            auto &creators = instance()->get_or_create_creator_list<Base>(type);
            creators[key] = []() -> std::shared_ptr<Base> {
                return std::make_shared<T>();
            };
        }

        // 创建 shared_ptr<Base> 对象
        template<typename Base>
        std::shared_ptr<Base> create_shared(const std::string &type, const std::string &key) {
            auto it = instance()->m_base_factories.find(type_id<Base>());
            if (it == instance()->m_base_factories.end()) return nullptr;

            // 错误：不能 cast 到指针！
            // auto &creators = *std::any_cast<std::unordered_map<...> *>(it->second);

            // 正确：cast 到引用
            auto &creators = std::any_cast<std::unordered_map<std::string, std::function<std::shared_ptr<Base>()>> &>(it->second);

            auto creator_it = creators.find(key);
            if (creator_it == creators.end()) return nullptr;
            return creator_it->second();
        }

        // 获取某类型下的所有创建器
        template<typename Base>
        std::unordered_map<std::string, std::function<std::shared_ptr<Base>()> > create_shared_list(
            const std::string &type) {
            auto it = instance()->m_base_factories.find(type_id<Base>());
            if (it == instance()->m_base_factories.end()) return {};

            // 同样，不能 cast 到指针
            // return *std::any_cast<std::map<...> *>(it->second);

            return std::any_cast<std::unordered_map<std::string, std::function<std::shared_ptr<Base>()>> &>(it->second);
        }

    private:
        // 为每个 Base 类型生成唯一 ID（避免 RTTI 依赖）
        template<typename T>
        static const std::type_info *type_id() { return &typeid(T); }

        // 获取或创建特定 Base 类型的 creator 映射
        template<typename Base>
        std::unordered_map<std::string, std::function<std::shared_ptr<Base>()> > &get_or_create_creator_list(
            const std::string &type) {
            const std::type_info *base_id = type_id<Base>();
            auto &any_map = m_base_factories[base_id];

            if (!any_map.has_value()) {
                any_map = std::make_any<std::unordered_map<std::string, std::function<std::shared_ptr<Base>()> > >();
            }

            return std::any_cast<std::unordered_map<std::string, std::function<std::shared_ptr<Base>()> > &>(any_map);
        }

    private:
        // 外层 map: type_name -> (map of creators for a specific Base type)
        std::unordered_map<const std::type_info *, std::any> m_base_factories;
    };

#define REGISTER_MODULE_OBJECT(T, Base,type, key, ...) \
    static struct reg_##T##_##__COUNTER__ { \
            reg_##T##_##__COUNTER__() { \
                ::tff::factory::ModuleFactory::register_type<T, Base>(type, key, ##__VA_ARGS__); \
            } \
    } reg_##T##_##__COUNTER__##_instance;

}
#endif // DEEP_TFF_MODULEFACTORY_H
