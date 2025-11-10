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
#include "mem/Memory.h"
#include "mem/Tensor.h"
#include "graph/Graph.h"
#include "graph/GraphNode.h"
#include "model/base/ModelLoaderBase.h"
namespace tff::core::model {
    class ModelCreatorBase {
    public:
        ModelCreatorBase() = default;

        virtual ~ModelCreatorBase() = default;

    public:
#define ADD_NODE(enum_op_type) \
        tff::factory::ModuleFactory::instance()->create_shared<tff::core::graph::GraphNode>(OP_NODE_FLAG,\
        tff::factory::ModuleKeyType(enum_op_type));

        using NodeType = std::unordered_map<tff::core::graph::GraphNodeType, std::shared_ptr<graph::GraphNode> >;

    public:
        virtual void build_layer(std::shared_ptr<tff::core::memory::Tensor> &tensor_ptr,
                                 std::shared_ptr<tff::core::graph::GraphNode> &layer_node,
                                 const size_t &total_layer_num = -1, const size_t &layer_index = -1) = 0;

        virtual void build_graph(std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                                     std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                         tff::core::graph::GraphNode> > > > &layer_map,
                                 std::shared_ptr<tff::core::graph::Graph> &graph_ptr) = 0;

        //
        virtual const char *get_model_name() = 0;
        //
        virtual void set_loader(const std::shared_ptr<tff::core::model::ModelLoaderBase> &loader) = 0;

        // public:
        //     static void register_create_layer() {
        //         auto callback = &Derived::create_layer;
        //         using callback_type = decltype(callback);
        //         // 子类必须有create_layer函数实现;
        //         static_assert(
        //             std::is_pointer_v<callback_type> &&
        //             std::is_function_v<std::remove_pointer_t<callback_type> >,
        //             "Derived::create_layer must be a function"
        //         );
        //
        //         //using traits = tff::core::global::FunctionTraits<callback_type>;
        //         //using signature = typename traits::type;
        //
        //         tff::factory::FunctionFactory::instance()->register_callback(
        //             CREATE_LAYER_FLAG,
        //             Derived::get_model_name(),
        //             callback
        //         );
        //     };
        //     static void register_create_graph() {
        //         auto callback = &Derived::build_graph;
        //         using callback_type = decltype(callback);
        //         // 子类必须有create_layer函数实现;
        //         static_assert(
        //             std::is_pointer_v<callback_type> &&
        //             std::is_function_v<std::remove_pointer_t<callback_type> >,
        //             "Derived::create_layer must be a function"
        //         );
        //
        //         //using traits = tff::core::global::FunctionTraits<callback_type>;
        //         //using signature = typename traits::type;
        //
        //         tff::factory::FunctionFactory::instance()->register_callback(
        //             BUILD_GRAPH_FLAG,
        //             Derived::get_model_name(),
        //             callback
        //         );
        //     }
        // public:
        //     static void registry_function() {
        //         //model layer creator;
        //         register_create_layer();
        //         //model graph builder;
        //         register_create_graph();
        //     }
    };
}
#endif //TFFINFER_MODELCREATORBASE_H
