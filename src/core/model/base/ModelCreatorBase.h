//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_MODELCREATORBASE_H
#define TFFINFER_MODELCREATORBASE_H
#include <functional>
#include <tuple>
#include "global/GlobalDefine.h"
#include "FunctionFactory.h"
#include "global/FunctionTraits.h"

namespace tff::core::model {
    template<typename Derived>
    class ModelCreatorBase {
    public:
        static void register_create_layer() {
            auto callback = &Derived::create_layer;
            using callback_type = decltype(callback);
            // 子类必须有create_layer函数实现;
            static_assert(
                std::is_pointer_v<callback_type> &&
                std::is_function_v<std::remove_pointer_t<callback_type> >,
                "Derived::create_layer must be a function"
            );

            using traits = tff::core::global::FunctionTraits<callback_type>;
            using signature = typename traits::type;

            tff::factory::FunctionFactory::instance()->register_callback(
                CREATE_LAYER_FLAG,
                Derived::get_model_name(),
                callback
            );
        };
        static void register_create_graph() {
            auto callback = &Derived::build_graph;
            using callback_type = decltype(callback);
            // 子类必须有create_layer函数实现;
            static_assert(
                std::is_pointer_v<callback_type> &&
                std::is_function_v<std::remove_pointer_t<callback_type> >,
                "Derived::create_layer must be a function"
            );

            using traits = tff::core::global::FunctionTraits<callback_type>;
            using signature = typename traits::type;

            tff::factory::FunctionFactory::instance()->register_callback(
                BUILD_GRAPH_FLAG,
                Derived::get_model_name(),
                callback
            );
        }
    public:
        static void registry_function() {
            //model layer creator;
            register_create_layer();
            //model graph builder;
            register_create_graph();
        }
    };
}
#endif //TFFINFER_MODELCREATORBASE_H
