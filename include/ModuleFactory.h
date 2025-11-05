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
#include <variant>

namespace tff::factory {
    using ModuleKeyType = std::variant<std::string, int>;
    struct ModuleKeyHash {
        using is_transparent = void;

        std::size_t operator()(const std::string &str) const {
            return std::hash<std::string>{}(str);
        }

        std::size_t operator()(int enum_val) const {
            return std::hash<int>{}(enum_val);
        }

        std::size_t operator()(const ModuleKeyType &key) const {
            return std::visit([this](const auto &k) { return (*this)(k); }, key);
        }
    };

    struct ModuleKeyEqual {
        using is_transparent = void;

        bool operator()(const std::string &lhs, const std::string &rhs) const {
            return lhs == rhs;
        }

        bool operator()(int lhs, int rhs) const {
            return lhs == rhs;
        }

        bool operator()(const ModuleKeyType &lhs, const ModuleKeyType &rhs) const {
            return lhs == rhs;
        }

        bool operator()(const std::string &lhs, const ModuleKeyType &rhs) const {
            return std::holds_alternative<std::string>(rhs) && lhs == std::get<std::string>(rhs);
        }

        bool operator()(const ModuleKeyType &lhs, const std::string &rhs) const {
            return std::holds_alternative<std::string>(lhs) && std::get<std::string>(lhs) == rhs;
        }

        bool operator()(int lhs, const ModuleKeyType &rhs) const {
            return std::holds_alternative<int>(rhs) && lhs == std::get<int>(rhs);
        }

        bool operator()(const ModuleKeyType &lhs, int rhs) const {
            return std::holds_alternative<int>(lhs) && std::get<int>(lhs) == rhs;
        }
    };

    class DEEP_TFF_API ModuleFactory {
    public:
        static std::shared_ptr<ModuleFactory> &instance();

    public:
        ModuleFactory(const ModuleFactory &) = delete;

        ModuleFactory &operator=(const ModuleFactory &) = delete;

        ModuleFactory(ModuleFactory &&) = delete;

        ModuleFactory &operator=(ModuleFactory &&) = delete;

    private:
        ModuleFactory() = default;

    public:
        template<typename T, typename Base, typename... Args>
        static void register_type(const std::string &type, const ModuleKeyType &key, Args &&... args) {
            auto &creators = instance()->get_or_create_creator_list<Base>();
            creators[key] = [args...]() -> std::shared_ptr<Base> {
                return std::make_shared<T>(args...);
            };
        }

        template<typename T, typename Base>
        static void register_type(const std::string &type, const ModuleKeyType &key) {
            auto &creators = instance()->get_or_create_creator_list<Base>();
            creators[key] = []() -> std::shared_ptr<Base> {
                return std::make_shared<T>();
            };
        }

        template<typename Base>
        std::shared_ptr<Base> create_shared(const std::string &type, const ModuleKeyType &key) {
            auto it = instance()->_base_factories.find(type_id<Base>());
            if (it == instance()->_base_factories.end()) return nullptr;
            using MapType = std::unordered_map<ModuleKeyType, std::function<std::shared_ptr<Base>()>,
                ModuleKeyHash, ModuleKeyEqual>;
            auto &creators = std::any_cast<MapType &>(it->second);
            auto creator_it = creators.find(key);
            if (creator_it == creators.end()) return nullptr;
            return creator_it->second();
        }

        template<typename Base>
        std::shared_ptr<Base> create_shared(const std::string &type, const std::string &key_string) {
            return create_shared<Base>(type, ModuleKeyType(key_string));
        }

        // 获取某类型下的所有创建器
        template<typename Base>
        std::unordered_map<ModuleKeyType, std::function<std::shared_ptr<Base>()>,ModuleKeyHash, ModuleKeyEqual > create_shared_list(
            const std::string &type) {
            auto it = instance()->_base_factories.find(type_id<Base>());
            if (it == instance()->_base_factories.end()) {
                return std::unordered_map<ModuleKeyType, std::function<std::shared_ptr<Base>()>,
                    ModuleKeyHash, ModuleKeyEqual>();
            }
            using MapType = std::unordered_map<ModuleKeyType, std::function<std::shared_ptr<Base>()>,
                ModuleKeyHash, ModuleKeyEqual>;
            auto &creators = std::any_cast<MapType &>(it->second);

            return creators;
        }

    private:
        template<typename T>
        static const std::type_info *type_id() { return &typeid(T); }

        template<typename Base>
        std::unordered_map<ModuleKeyType, std::function<std::shared_ptr<Base>()>, ModuleKeyHash, ModuleKeyEqual> &
        get_or_create_creator_list() {
            const std::type_info *base_id = type_id<Base>();
            auto &any_map = _base_factories[base_id];
            if (!any_map.has_value()) {
                any_map = std::make_any<std::unordered_map<ModuleKeyType, std::function<std::shared_ptr<Base>()>,
                    ModuleKeyHash, ModuleKeyEqual> >();
            }

            using MapType = std::unordered_map<ModuleKeyType, std::function<std::shared_ptr<Base>()>, ModuleKeyHash,
                ModuleKeyEqual>;

            return std::any_cast<MapType &>(any_map);
        }

    private:
        //std::unordered_map<const std::type_info *, std::any> m_base_factories;
        std::unordered_map<
            const std::type_info *,
            std::any,
            std::hash<const std::type_info *>,
            std::equal_to<const std::type_info *> > _base_factories;
    };

#define REGISTER_MODULE_OBJECT(T, Base,type, key, ...) \
    static struct reg_##T##_##__COUNTER__ { \
            reg_##T##_##__COUNTER__() { \
                ::tff::factory::ModuleFactory::register_type<T, Base>(type, key, ##__VA_ARGS__); \
            } \
    } reg_##T##_##__COUNTER__##_instance;
}
#endif // DEEP_TFF_MODULEFACTORY_H
