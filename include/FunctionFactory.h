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
#include "Logger.h"

namespace tff::factory {
    /**
     * 函数工厂
     */
    class DEEP_TFF_API FunctionFactory : public std::enable_shared_from_this<FunctionFactory> {
    public:
        static std::shared_ptr<FunctionFactory> instance() {
            static std::shared_ptr<FunctionFactory> s_instance = create_instance();
            return s_instance;
        }

    public:
        /**
         * 注册回调
         * @tparam Func
         * @param flag
         * @param key
         * @param func
         */
        template<typename Func>
        void register_callback(const std::string &flag, const std::string &key, Func &&func) {
            std::lock_guard<std::mutex> lock(_mutex);
            _functions[flag][key] = std::function(func);
        }
        /**
         * 获取回调
         * @tparam Func
         * @param flag
         * @param key
         * @return
         */
        template<typename Func>
        std::function<Func> get_callback(const std::string &flag, const std::string &key) {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _functions.find(flag);
            if (it == _functions.end()) {
                tff::log::Logger::error("Flag not found");
                return nullptr;
            }
            auto it2 = it->second.find(key);
            if (it2 == it->second.end()) {
                tff::log::Logger::error("callback type: %s key type: %s Callback not found", flag.c_str(), key.c_str());
                return nullptr;
            }
            return std::any_cast<std::function<Func> >(it2->second);
        }
        /**
         * 检测回调是否存在
         * @param flag
         * @param key
         * @return
         */
        bool has_callback(const std::string &flag, const std::string &key) {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _functions.find(flag);
            if (it == _functions.end()) return false;
            return it->second.find(key) != it->second.end();
        }
        /**
         * 调用回调
         * @tparam Args
         * @param flag
         * @param key
         * @param args
         */
        template<typename... Args>
        void invoke(const std::string &flag, const std::string &key, Args &&... args) {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _functions.find(flag);
            if (it == _functions.end()) tff::log::Logger::error("Flag not found");
            auto it2 = it->second.find(key);
            if (it2 == it->second.end()) tff::log::Logger::error("Callback not found");
            std::any_cast<std::function<void(Args...)> >(it2->second)(std::forward<Args>(args)...);
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
        std::unordered_map<std::string, std::unordered_map<std::string, std::any> > _functions;

        mutable std::mutex _mutex; // 保证线程安全
    };
} // namespace TFF
#endif // DEEP_TFF_FUNCTIONFACTORY_H
