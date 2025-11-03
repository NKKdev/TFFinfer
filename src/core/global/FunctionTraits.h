//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_FUNCTIONTRAITS_H
#define TFFINFER_FUNCTIONTRAITS_H

namespace tff::core::global {
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
}

#endif //TFFINFER_FUNCTIONTRAITS_H
