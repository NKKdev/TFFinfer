//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_MODELCREATORBASE_H
#define TFFINFER_MODELCREATORBASE_H
#include <functional>
#include <tuple>
#include "global/GlobalDefine.h"
#include "FunctionFactory.h"

namespace tff::core::model {
    template<typename T>
    struct FunctionTraits;

    // 函数指针
    template<typename Ret, typename... Args>
    struct FunctionTraits<Ret(*)(Args...)> {
        using type = Ret(Args...);
        using return_type = Ret;
        using args_tuple = std::tuple<Args...>;
        static constexpr size_t arity = sizeof...(Args);
    };

    // std::function
    template<typename Ret, typename... Args>
    struct FunctionTraits<std::function<Ret(Args...)> >
            : FunctionTraits<Ret(Args...)> {
    };

    // 成员函数
    template<typename Class, typename Ret, typename... Args>
    struct FunctionTraits<Ret(Class::*)(Args...)> {
        using type = Ret(Args...);
        using return_type = Ret;
        using args_tuple = std::tuple<Args...>;
        static constexpr size_t arity = sizeof...(Args);
    };

    // const 成员函数
    template<typename Class, typename Ret, typename... Args>
    struct FunctionTraits<Ret(Class::*)(Args...) const> {
        using type = Ret(Args...);
        using return_type = Ret;
        using args_tuple = std::tuple<Args...>;
        static constexpr size_t arity = sizeof...(Args);
    };

    //lambda
    template<typename T>
    struct FunctionTraits : public FunctionTraits<decltype(&T::operator())> {
    };

    template<typename Derived>
    class ModelCreatorBase {
    public:
        static void registry_function() {
            auto callback = &Derived::create_layer;
            using callback_type = decltype(callback);
            // 子类必须有create_layer函数实现;
            static_assert(
                std::is_pointer_v<callback_type> &&
                std::is_function_v<std::remove_pointer_t<callback_type> >,
                "Derived::create_layer must be a function"
            );

            using traits = FunctionTraits<callback_type>;
            using signature = typename traits::type;

            tff::factory::FunctionFactory::instance()->register_callback(
                CREATE_LAYER_FLAG,
                Derived::get_model_name(),
                callback
            );
        }
    };
}
#endif //TFFINFER_MODELCREATORBASE_H
