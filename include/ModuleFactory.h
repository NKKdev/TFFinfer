//
// Created by nkk on 2025/4/27.
// Modified to support both Singleton and Prototype creation policies.
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
#include <mutex>

#include "Logger.h"

namespace tff::factory {
    using ModuleKeyType = std::variant<std::string, int>;
    /**
     * ModuleKeyHash and ModuleKeyEqual are used to allow ModuleKeyType to be used as a key in an unordered_map.
     */
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
    /**
     * ModuleKeyHash and ModuleKeyEqual are used to allow ModuleKeyType to be used as a key in an unordered_map.
     */
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

    enum class CreationPolicy {
        PROTOTYPE,
        SINGLETON
    };

    template<typename Base>
    struct CreatorInfo {
        std::function<std::shared_ptr<Base>()> creator;
        CreationPolicy policy;
    };
    /**
     * @brief 模块工厂
     */
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
        /**
         * @brief 模块工厂单例
         */
        template<typename Base>
        struct SingletonInstanceMap {
            static std::unordered_map<ModuleKeyType, std::weak_ptr<Base>, ModuleKeyHash, ModuleKeyEqual> instances;
            //static std::mutex mtx;
        };
        /**
         * @brief 获取单例实例
         */
        template<typename Base>
        std::shared_ptr<Base> get_singleton_instance(const ModuleKeyType &key,
                                                     const std::function<std::shared_ptr<Base>()> &creator) {
            auto &map = SingletonInstanceMap<Base>::instances;
            //auto &mtx = SingletonInstanceMap<Base>::mtx;

            //std::lock_guard<std::mutex> lock(mtx);
            auto it = map.find(key);
            if (it != map.end()) {
                std::shared_ptr<Base> ptr = it->second.lock();
                if (ptr) {
                    return ptr;
                } else {
                    map.erase(it);
                }
            }

            std::shared_ptr<Base> new_instance = creator();
            map[key] = new_instance;
            return new_instance;
        }

    public:
        /**
         * @brief 注册类型
         * @param  type  类型
         * @param  key   键
         * @param  args   参数
         */
        template<typename T, typename Base, typename... Args>
        static void register_type(const std::string &type, const ModuleKeyType &key, Args &&... args) {
            register_type_with_policy<T, Base>(type, key, CreationPolicy::PROTOTYPE, std::forward<Args>(args)...);
        }
        /**
         * @brief 注册类型
         * @param  type  类型
         * @param  key   键
         */
        template<typename T, typename Base>
        static void register_type(const std::string &type, const ModuleKeyType &key) {
            register_type_with_policy<T, Base>(type, key, CreationPolicy::PROTOTYPE);
        }

        /**
         *
         * @tparam T   子类类型
         * @tparam Base 基类
         * @tparam Args 参数
         * @param type 模块类型
         * @param key 键
         * @param policy 创建策略
         * @param args 参数
         */
        template<typename T, typename Base, typename... Args>
        static void register_type_with_policy(const std::string &type, const ModuleKeyType &key, CreationPolicy policy,
                                              Args &&... args) {
            auto &creators = instance()->get_or_create_creator_list<Base>(type);
            creators[key] = CreatorInfo<Base>{
                [args...]() -> std::shared_ptr<Base> {
                    return std::make_shared<T>(args...);
                },
                policy
            };
        }
        /**
         * @brief 注册类型
         * @param  type  类型
         * @param  key   键
         * @param  policy   创建策略
         */
        template<typename T, typename Base>
        static void register_type_with_policy(const std::string &type, const ModuleKeyType &key,
                                              CreationPolicy policy) {
            auto &creators = instance()->get_or_create_creator_list<Base>(type);
            creators[key] = CreatorInfo<Base>{
                []() -> std::shared_ptr<Base> {
                    return std::make_shared<T>();
                },
                policy
            };
        }

    public:
        /**
         * @brief 创建对象
         * @param  type  类型
         * @param  key   键
         * @return  对象
         */
        template<typename Base>
        std::shared_ptr<Base> create_shared(const std::string &type, const ModuleKeyType &key) {
            auto it = instance()->_typed_factories.find(type);
            if (it != instance()->_typed_factories.end()) {
                using MapType = std::unordered_map<ModuleKeyType, CreatorInfo<Base>, ModuleKeyHash, ModuleKeyEqual>;
                try {
                    auto &creators = std::any_cast<MapType &>(it->second._creators);
                    auto creator_it = creators.find(key);
                    if (creator_it != creators.end()) {
                        const auto &info = creator_it->second;
                        if (info.policy == CreationPolicy::PROTOTYPE) {
                            return info.creator();
                        } else if (info.policy == CreationPolicy::SINGLETON) {
                            return instance()->get_singleton_instance<Base>(key, info.creator);
                        }
                    }
                } catch (const std::bad_any_cast &e) {
                    tff::log::Logger::error("create_shared: bad_any_cast %s", e.what());
                }
            }

            return nullptr;
        }
        /**
         * @brief 创建对象
         * @param  type  类型
         * @param  key   键
         * @return  对象
         */
        template<typename Base>
        std::shared_ptr<Base> create_shared(const std::string &type, const std::string &key_string) {
            return create_shared<Base>(type, ModuleKeyType(key_string));
        }
        /**
         * @brief 创建对象列表
         * @param  type  类型
         * @return  对象列表
         */
        template<typename Base>
        std::unordered_map<ModuleKeyType, CreatorInfo<Base>, ModuleKeyHash, ModuleKeyEqual>
        create_shared_list(const std::string &type_string) {
            const auto it = this ->_typed_factories.find(type_string);
            if (it != this->_typed_factories.end()) {
                using ExpectedMapType = std::unordered_map<ModuleKeyType, CreatorInfo<Base>, ModuleKeyHash,
                    ModuleKeyEqual>;
                try {
                    const auto &creators_any = it->second._creators;
                    const auto &creators_map = std::any_cast<const ExpectedMapType &>(creators_any);
                    return creators_map;
                } catch (const std::bad_any_cast &e) {
                    tff::log::Logger::error("create_shared_list creator type %s not found error: %s",
                                            type_string.c_str(), e.what());
                }
            }
            return std::unordered_map<ModuleKeyType, CreatorInfo<Base>, ModuleKeyHash, ModuleKeyEqual>();
        }

    private:
        /**
         * @brief 获取类型ID
         * @tparam T   类型
         * @return  类型ID
         */
        template<typename T>
        static const std::type_info *type_id() { return &typeid(T); }

        struct FactoryEntry {
            std::any _creators;
            std::string _category;
        };

        std::unordered_map<
            std::string,
            FactoryEntry,
            std::hash<std::string>,
            std::equal_to<std::string>
        > _typed_factories;
        /**
         * @brief 获取类型列表
         * @return  类型列表
         */
        template<typename Base>
        std::unordered_map<ModuleKeyType, CreatorInfo<Base>, ModuleKeyHash, ModuleKeyEqual> &
        get_or_create_creator_list(const std::string &type_category) {
            auto it = _typed_factories.find(type_category);
            if (it == _typed_factories.end()) {
                FactoryEntry entry;
                entry._creators = std::make_any<std::unordered_map<ModuleKeyType, CreatorInfo<Base>, ModuleKeyHash,
                    ModuleKeyEqual> >();
                entry._category = type_category;
                _typed_factories.emplace(type_category, std::move(entry));
                it = _typed_factories.find(type_category);
            }

            using MapType = std::unordered_map<ModuleKeyType, CreatorInfo<Base>, ModuleKeyHash, ModuleKeyEqual>;
            try {
                return std::any_cast<MapType &>(it->second._creators);
            } catch (const std::bad_any_cast &) {
                tff::log::Logger::error("Internal error: Type mismatch when retrieving creator list.");
            }
        }
    };

    template<typename Base>
    std::unordered_map<ModuleKeyType, std::weak_ptr<Base>, ModuleKeyHash, ModuleKeyEqual>
    ModuleFactory::SingletonInstanceMap<Base>::instances;

    // template<typename Base>
    // std::mutex ModuleFactory::SingletonInstanceMap<Base>::mtx;


#define REGISTER_MODULE_OBJECT_PROTOTYPE(T, Base, type, key, ...) \
        static struct reg_##T##_##__COUNTER__ { \
                reg_##T##_##__COUNTER__() { \
                    ::tff::factory::ModuleFactory::register_type<T, Base>(type, key, ##__VA_ARGS__); \
                } \
        } reg_##T##_##__COUNTER__##_instance;

#define REGISTER_MODULE_OBJECT(T, Base, type, key, ...) \
        static struct reg_singleton_##T##_##__COUNTER__ { \
                reg_singleton_##T##_##__COUNTER__() { \
                    ::tff::factory::ModuleFactory::register_type_with_policy<T, Base>(type, key, ::tff::factory::CreationPolicy::SINGLETON, ##__VA_ARGS__); \
                } \
        } reg_singleton_##T##_##__COUNTER__##_instance;
} // namespace tff::factory

#endif // DEEP_TFF_MODULEFACTORY_H
