//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_FUNCTIONTRAITS_H
#define TFFINFER_FUNCTIONTRAITS_H

namespace tff::core::global {
    /**
     * 函数特征
     * @tparam T
     */
    template<typename T>
    struct FunctionTraits;
    
    template<typename Ret, typename... Args>
    struct FunctionTraits<Ret(*)(Args...)> {
        using type = Ret(Args...);
        using return_type = Ret;
        using args_tuple = std::tuple<Args...>;
        static constexpr size_t arity = sizeof...(Args);
        template <size_t Index>
        struct argument {
            using type = typename std::tuple_element<Index, args_tuple>::type;
        };
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

        template <size_t Index>
        struct argument {
            using type = typename std::tuple_element<Index, args_tuple>::type;
        };
    };

    // const 成员函数
    template<typename Class, typename Ret, typename... Args>
    struct FunctionTraits<Ret(Class::*)(Args...) const> {
        using type = Ret(Args...);
        using return_type = Ret;
        using args_tuple = std::tuple<Args...>;
        static constexpr size_t arity = sizeof...(Args);

        template <size_t Index>
        struct argument {
            using type = typename std::tuple_element<Index, args_tuple>::type;
        };
    };
    //
    template<typename Ret, typename... Args>
    struct FunctionTraits<Ret(*&)(Args...)> : FunctionTraits<Ret(*)(Args...)> { };
    //
    template<typename Ret, typename... Args>
    struct FunctionTraits<Ret(*&&)(Args...)> : FunctionTraits<Ret(*)(Args...)> { };

    //lambda
    template<typename T>
    struct FunctionTraits : public FunctionTraits<decltype(&T::operator())> {
    };
}

#endif //TFFINFER_FUNCTIONTRAITS_H
