//
// Created by nkk on 2025/7/1.
//

#ifndef DEEP_TFF_FUNCTIONFACTORY_H
#define DEEP_TFF_FUNCTIONFACTORY_H
#include "ExportInc.h"
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <any>
#include <utility>

namespace tff::factory {
        class DEEP_TFF_API FunctionFactory : public std::enable_shared_from_this<FunctionFactory> {
        public:
            static std::shared_ptr<FunctionFactory> instance() {
                static std::shared_ptr<FunctionFactory> s_instance = create_instance();
                return s_instance;
            }

        public:
            // 注册一个函数：指定分组 group 和键 key
            template<typename R, typename... Args>
            void register_callback(const std::string &group,
                                   const std::string &key,
                                   std::function<R(Args...)> callback) {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto &group_map = m_functions[std::type_index(typeid(std::function<R(Args...)>))];
                group_map[group][key] = callback;
            }

            // 调用已注册的函数
            template<typename R, typename... Args>
            R invoke_callback(const std::string &group,
                              const std::string &key,
                              Args... args) {
                std::lock_guard<std::mutex> lock(m_mutex);

                auto type_it = m_functions.find(std::type_index(typeid(std::function<R(Args...)>)));
                if (type_it != m_functions.end()) {
                    const auto &group_map = type_it->second;
                    auto group_it = group_map.find(group);
                    if (group_it != group_map.end()) {
                        const auto &func_map = group_it->second;
                        auto func_it = func_map.find(key);
                        if (func_it != func_map.end()) {
                            try {
                                auto func = std::any_cast<std::function<R(Args...)> >(func_it->second);
                                return func(args...);
                            } catch (const std::bad_any_cast &) {
                                // 可选：添加日志
                                // Logger::error("Bad any_cast when invoking %s/%s", group.c_str(), key.c_str());
                            }
                        }
                    }
                }
                return nullptr;
            }

            // 检查函数是否存在
            template<typename FuncType>
            bool has_callback(const std::string &group, const std::string &key) {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto type_idx = std::type_index(typeid(FuncType));
                auto type_it = m_functions.find(type_idx);
                if (type_it != m_functions.end()) {
                    auto group_it = type_it->second.find(group);
                    if (group_it != type_it->second.end()) {
                        return group_it->second.find(key) != group_it->second.end();
                    }
                }
                return false;
            }

            // 移除某个函数
            template<typename FuncType>
            bool unregister_callback(const std::string &group, const std::string &key) {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto type_idx = std::type_index(typeid(FuncType));
                auto type_it = m_functions.find(type_idx);
                if (type_it != m_functions.end()) {
                    auto &group_map = type_it->second;
                    auto group_it = group_map.find(group);
                    if (group_it != group_map.end()) {
                        auto &func_map = group_it->second;
                        return func_map.erase(key) > 0;
                    }
                }
                return false;
            }

        private:
            FunctionFactory() = default;
            static std::shared_ptr<FunctionFactory> create_instance() {
                struct Constructor {
                    std::shared_ptr<FunctionFactory> operator()() {
                        return std::shared_ptr<FunctionFactory>(new FunctionFactory());
                    }
                };
                return Constructor()();
            }

        private:
            // 删除拷贝和移动构造
            FunctionFactory(const FunctionFactory &) = delete;

            FunctionFactory(FunctionFactory &&) = delete;

            FunctionFactory &operator=(const FunctionFactory &) = delete;

            FunctionFactory &operator=(FunctionFactory &&) = delete;

        private:
            // 三层结构：类型 -> group -> key -> function
            std::unordered_map<
                std::type_index,
                std::unordered_map<std::string,std::unordered_map<std::string,std::any>>> m_functions;

            mutable std::mutex m_mutex; // 保证线程安全
        };
} // namespace TFF
#endif // DEEP_TFF_FUNCTIONFACTORY_H
