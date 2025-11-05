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
#include "global/FunctionTraits.h"

namespace tff::factory {
    // FunctionFactory.h
    using GenericArgs = std::vector<std::any>;

    class InvokableBase {
    public:
        virtual ~InvokableBase() = default;

        virtual std::any invoke_generic(GenericArgs &args) = 0;
    };

    template<typename Func>
    class Invokable : public InvokableBase {
    private:
        Func _func;

        template<std::size_t Index>
        decltype(auto) extract_argument(const std::any &any_arg) {
            using traits = tff::core::global::FunctionTraits<Func>;
            using ArgTypeDeclared = typename traits::template argument<Index>::type;
            using ArgTypeDecayed = std::decay_t<ArgTypeDeclared>;

            tff::log::Logger::info("Arg %zu: declared=%s, decayed=%s, any_type=%s",
                                   Index,
                                   typeid(ArgTypeDeclared).name(),
                                   typeid(ArgTypeDecayed).name(),
                                   any_arg.type().name());

            if constexpr (std::is_lvalue_reference_v<ArgTypeDeclared> &&
                          !std::is_const_v<std::remove_reference_t<ArgTypeDeclared> >) {
                auto rw = std::any_cast<std::reference_wrapper<ArgTypeDecayed> >(any_arg);
                tff::log::Logger::info("Extracted reference to value: %d", rw.get());
                return rw.get();
            } else {
                auto val = std::any_cast<ArgTypeDecayed>(any_arg);
                tff::log::Logger::info("Extracted value: %d", val);
                return val;
            }
        }

        template<std::size_t... Is>
        std::any call_with_indices_from_vector(const GenericArgs &args_vector, std::index_sequence<Is...>) {
            using traits = tff::core::global::FunctionTraits<Func>;
            constexpr std::size_t arg_count = traits::arity;
            if (args_vector.size() != arg_count) {
                tff::log::Logger::error("Argument count mismatch for callback.");
                return {};
            }
            try {
                auto args_tuple = std::make_tuple(extract_argument<Is>(args_vector[Is])...);
                if constexpr (std::is_void_v<typename traits::return_type>) {
                    //std::apply(_func, std::move(args_tuple));
                    _func(extract_argument<Is>(args_vector[Is])...);
                    return std::any{};
                } else {
                    return std::apply(_func, std::move(args_tuple));
                }
            } catch (const std::bad_any_cast &e) {
                tff::log::Logger::error("Type mismatch in arguments: %s", e.what());
            } catch (const std::exception &e) {
                tff::log::Logger::error("Exception in callback: %s", e.what());
            } catch (...) {
                tff::log::Logger::error("Unknown exception in callback.");
            }
            return {};
        }

        template<typename... CallArgs>
        std::any call_direct(CallArgs &&... args) {
            using traits = tff::core::global::FunctionTraits<Func>;
            if constexpr (std::is_void_v<typename traits::return_type>) {
                _func(std::forward<CallArgs>(args)...);
                return std::any{};
            } else {
                return _func(std::forward<CallArgs>(args)...);
            }
        }

    public:
        explicit Invokable(Func &&f) : _func(std::forward<Func>(f)) {
        }

        std::any invoke_generic(GenericArgs &args) override {
            constexpr std::size_t arg_count = tff::core::global::FunctionTraits<Func>::arity;
            return call_with_indices_from_vector(args, std::make_index_sequence<arg_count>{});
        }

        template<typename... CallArgs>
        std::any invoke(CallArgs &&... args) {
            return call_direct(std::forward<CallArgs>(args)...);
        }
    };

    class DEEP_TFF_API FunctionFactory : public std::enable_shared_from_this<FunctionFactory> {
    public:
        static std::shared_ptr<FunctionFactory> instance() {
            static std::shared_ptr<FunctionFactory> s_instance = create_instance();
            return s_instance;
        }

    public:
        template<typename Func>
        void register_callback(const std::string &flag, const std::string &key, Func &&func) {
            std::lock_guard<std::mutex> lock(_mutex);
            auto wrapper = create_wrapper(std::forward<Func>(func));
            _functions[flag][key] = wrapper;
        }

        std::shared_ptr<InvokableBase> get_callback(const std::string &flag, const std::string &key) const {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _functions.find(flag);
            if (it != _functions.end()) {
                auto it2 = it->second.find(key);
                if (it2 != it->second.end()) {
                    try {
                        return std::any_cast<std::shared_ptr<InvokableBase> >(it2->second);
                    } catch (const std::bad_any_cast &e) {
                        tff::log::Logger::error("Bad any_cast for callback base pointer.");
                    }
                }
            }
            return nullptr;
        }

        // 判断是否存在
        bool has_callback(const std::string &flag, const std::string &key) {
            std::lock_guard<std::mutex> lock(_mutex);
            auto it = _functions.find(flag);
            if (it == _functions.end()) return false;
            return it->second.find(key) != it->second.end();
        }

        //
        std::any invoke_generic(const std::string &flag, const std::string &key, GenericArgs &args) {
            std::lock_guard<std::mutex> lock(_mutex);

            auto it = _functions.find(flag);

            if (it == _functions.end()) {
                tff::log::Logger::error("Flag not found:%s ", flag.c_str());
                return {};
            }

            auto it2 = it->second.find(key);

            if (it2 == it->second.end()) {
                tff::log::Logger::error("Callback not found for key:%s under flag:%s", key.c_str(), flag.c_str());
                return {};
            }


            try {
                auto base_ptr = std::any_cast<std::shared_ptr<InvokableBase> >(it2->second);

                if (!base_ptr) {
                    tff::log::Logger::error("Stored callback is invalid for key: %s ", key.c_str());
                }

                return base_ptr->invoke_generic(args);
            } catch (const std::bad_any_cast &e) {
                tff::log::Logger::error("Bad any_cast when retrieving callback for key: %s", key.c_str());
            }
            return {};
        }

        template<typename... Args>
        std::any invoke(const std::string &flag, const std::string &key, Args &&... args) {
            GenericArgs generic_args;
            generic_args.reserve(sizeof...(Args));
            (generic_args.emplace_back(std::forward<Args>(args)), ...);
            return this->invoke_generic(flag, key, generic_args);
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

        //
        template<typename T>
        std::shared_ptr<InvokableBase> create_wrapper(T &&func) {
            auto invokable_obj = std::make_shared<Invokable<T> >(std::forward<T>(func));
            return std::static_pointer_cast<InvokableBase>(invokable_obj);
        }

    private:
        // 删除拷贝和移动构造
        FunctionFactory(const FunctionFactory &) = delete;

        FunctionFactory(FunctionFactory &&) = delete;

        FunctionFactory &operator=(const FunctionFactory &) = delete;

        FunctionFactory &operator=(FunctionFactory &&) = delete;

    private:
        std::unordered_map<std::string, std::unordered_map<std::string, std::any> > _functions;

        mutable std::mutex _mutex; // 保证线程安全
    };
} // namespace TFF
#endif // DEEP_TFF_FUNCTIONFACTORY_H
