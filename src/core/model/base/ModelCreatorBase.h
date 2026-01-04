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
#include "model/layer/ModelLayer.h"
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
        virtual void build_graph(std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
                                     std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                                         tff::core::model::layer::ModelLayerObject> > > > &layer_map,
                                 std::shared_ptr<tff::core::graph::Graph> &graph_ptr) = 0;

        //
        virtual const char *get_model_name() = 0;

        //
        virtual void set_loader(const std::shared_ptr<tff::core::model::ModelLoaderBase> &loader) = 0;
    };
}
#endif //TFFINFER_MODELCREATORBASE_H
