//
// Created by nkk on 2/10/26.
//

#include "MemOptStrategy.h"

#include <taskflow/core/observer.hpp>

#include "include/TFFOPCreator.h"

namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(MemOptStrategyBase, tff::module::ModuleObject, MEM_OPT_STRATEGY_FLAG,
                               factory::ModuleKeyType(to_string(TFF_MEM_OPT_STRATEGY_DEFAULT)));

    std::shared_ptr<core::graph::GraphNode> MemOptStrategyBase::mem_optimize(
        const std::shared_ptr<graph::GraphNode> &node) {
        if (node == nullptr || node->op_type() == graph::TFF_OP_VIEW ||
            node->op_type() == graph::TFF_OP_RESHAPE ||
            !this->should_optimize(node)) {
            return nullptr;
        }
        auto mem_opt_node = ADD_NODE(core::graph::TffOpType::TFF_OP_MEM_RECYCLE);
        mem_opt_node->set_node_meta(graph::NodeMetadata{"mem_recycle_node"});
        const auto &builder =
            std::dynamic_pointer_cast<kernel::MemOptOPBuilder>(mem_opt_node->builder());
        builder->in(node->get_tensor());
        mem_opt_node->shape_infer();
        mem_opt_node->add_input_node(node);
        for (const auto &consumer : node->output_nodes()) {
            mem_opt_node->set_layer_id(consumer->layer_id());
            consumer->remove_input_node(node);
            consumer->add_input_node(mem_opt_node);
        }
        return mem_opt_node;
    }

    bool tff::core::runtime::MemOptStrategyBase::should_optimize(const std::shared_ptr<core::graph::GraphNode> &node) {
        if (node->get_tensor()->get_allocator()->_device_id == -1 ||  node->is_leaf()) {
            return false;
        }
        bool bRet = true;
        for (const auto &input : node->input_nodes()) {
            if (input->layer_id() == node->layer_id() || input->is_output_node()) {
                bRet &= false;
            }
        }
        return bRet;
    }
}
