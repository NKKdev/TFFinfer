//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_TFFOPCREATORBASE_H
#define TFFINFER_TFFOPCREATORBASE_H
#include "core/global/FunctionTraits.h"
#include "FunctionFactory.h"
#include "core/global/GlobalDefine.h"
namespace tff::kernel::base {
    template<typename Derived, typename T>
    class OPCreatorBase {
    public:
        static void registry_function() {
            auto callback = &Derived::compute;
            using callback_type = decltype(callback);
            // 子类必须有compute函数实现;
            static_assert(
                std::is_pointer_v<callback_type> &&
                std::is_function_v<std::remove_pointer_t<callback_type> >,
                "Derived::compute must be a function"
            );

            //using traits = tff::core::global::FunctionTraits<callback_type>;
            //using signature = typename traits::type;

            tff::factory::FunctionFactory::instance()->register_callback(
                OP_NODE_FLAG,
                Derived::get_op_name(),
                callback
            );
        }
    };
}
#define REGISTER_OP_OBJECT(T, type) \
namespace { \
    struct reg_##T##_##type##_##__LINE__ { \
        reg_##T##_##type##_##__LINE__() { \
            ::tff::kernel::T<type>::registry_function(); \
        } \
    } reg_##T##_##type##_##__LINE__##_instance; \
}
#endif //TFFINFER_TFFOPCREATORBASE_H
