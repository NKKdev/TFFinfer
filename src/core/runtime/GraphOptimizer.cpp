//
// Created by nkk on 2026/2/18.
//

#include "GraphOptimizer.h"

#include <ranges>

#include "MemOptStrategy.h"
#include "device/DeviceManager.h"
#include "global/ModelGlobalVar.h"
#include "graph/Graph.h"
#include "graph/GraphNode.h"
#include "include/TFFOPCreator.h"
#include "model/base/ModelCreatorBase.h"

namespace tff::core::runtime {
    REGISTER_MODULE_OBJECT(GraphOptimizer, tff::module::ModuleObject, GRAPH_OPTIMIZER_FALG, GRAPH_OPTIMIZER_FALG);

    void GraphOptimizer::optimize(model::GraphContext &graph_ctx,std::shared_ptr<core::graph::Graph> &graph_ptr) {
        tff::log::Logger::info("GraphOptimizer::optimize");
        //设备分配优化
        device_placement_opt(graph_ctx, graph_ptr);
        //算子融合
        fuse(graph_ptr);
        //显存优化
        mem_opt(graph_ptr);
        //
        //elimination_dead_code(graph_ptr);

        graph_ptr->build_graph(graph_ptr->output()[0]);
        tff::log::Logger::info("GraphOptimizer::optimize end");
    }

    void GraphOptimizer::device_placement_opt(model::GraphContext &graph_ctx,std::shared_ptr<core::graph::Graph> &graph_ptr) {
        if (!graph_ctx._layer_device_map.empty()) {
            graph_ctx._layer_device_map.clear();
        }
        auto device_manager = std::dynamic_pointer_cast<device::DeviceManager>(
            tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                DEVICE_MANAGER_FLAG,
                tff::factory::ModuleKeyType(DEVICE_MANAGER_FLAG)));

        int cur_device_id = -1;
        for (const auto &node: graph_ptr->total_nodes()) {
            if (node->op_type() == graph::TffOpType::TFF_OP_MAP2CPU ||
                node->op_type() == graph::TffOpType::TFF_OP_MEM_REF ||
                node->op_type() == graph::TffOpType::TFF_OP_VIEW) {
                continue;
            }
            const auto &node_device_id = node->get_tensor()->get_allocator()->_device_id;
            if (node_device_id != -1) {
                cur_device_id = node_device_id;
            } else if (cur_device_id != -1) {
                if (!node->is_input_node()&& !node->is_output_node()) {
                    auto allocator =
                            device_manager->get_device(cur_device_id)->get_device_buffer_allocator(cur_device_id);
                    node->get_tensor()->set_allocator(allocator);
                }
            }
        }
        //
        cur_device_id = -1;
        for (const auto &node: std::views::reverse(graph_ptr->total_nodes())) {
            if (node->op_type() == graph::TffOpType::TFF_OP_MAP2CPU ||
                node->op_type() == graph::TffOpType::TFF_OP_MEM_REF ||
                node->op_type() == graph::TffOpType::TFF_OP_VIEW) {
                continue;
            }
            const auto &node_device_id = node->get_tensor()->get_allocator()->_device_id;
            if (node_device_id != -1) {
                cur_device_id = node_device_id;
            } else if (cur_device_id != -1) {
                if (!node->is_input_node() && !node->is_output_node()) {
                    auto allocator =
                            device_manager->get_device(cur_device_id)->get_device_buffer_allocator(cur_device_id);
                    node->get_tensor()->set_allocator(allocator);
                }
            }
        }
        //
        for (const auto &node: graph_ptr->total_nodes()) {
            const auto &allocator = node->get_tensor()->get_allocator();
            if (allocator == nullptr) {
                continue;
            }
            const int &device_id = allocator->_device_id;
            if (node->op_type() == graph::TffOpType::TFF_OP_SET_ROWS) {//更新kv cache IO 节点设备
                graph_ctx._layer_device_map[node->layer_id()] = device_id;
            }
            for (const auto &input_node: node->input_nodes()) {
                const auto &input_allocator =
                        input_node->get_tensor()->get_allocator();
                if (input_allocator == nullptr) {
                    continue;
                }
                const int &input_device_id = input_allocator->_device_id;
                if (device_id != input_device_id) {
                    if (input_node->op_type() == graph::TffOpType::TFF_OP_MAP2CPU ||
                        input_node->op_type() == graph::TffOpType::TFF_OP_MEM_REF ||
                        input_node->op_type() == graph::TffOpType::TFF_OP_VIEW) {
                        continue;
                    }
                    auto src_type = input_allocator->device_type();
                    auto dst_type = allocator->device_type();
                    auto mem_cpy_node = ADD_NODE(graph::TffOpType::TFF_OP_MEM_CPY);
                    mem_cpy_node->set_node_meta(graph::NodeMetadata{"mem_cpy_node"});
                    const auto &builder =
                            std::dynamic_pointer_cast<kernel::MemCpyBuilder>(mem_cpy_node->builder());
                    builder->in(input_node->get_tensor())
                            .memcpy_kind(make_cpy_kind(src_type, dst_type))
                            .source_id(input_device_id)
                            .dest_id(device_id);
                    mem_cpy_node->shape_infer();
                    input_node->remove_output_node(node);

                    insert_input_node(node, mem_cpy_node, input_node);
                }
            }
        }
    }

    void GraphOptimizer::elimination_dead_code(std::shared_ptr<core::graph::Graph> &graph_ptr) {
        auto nodes = graph_ptr->nodes();
        for (auto &node: nodes) {
            if (node->is_fuse()) {
                graph_ptr->remove_node(node);
            }
        }
    }

    void GraphOptimizer::mem_opt(std::shared_ptr<core::graph::Graph> &graph_ptr) {
        auto mem_opt_strategy = MemOptStrategyBase::get_mem_opt_strategy();
        for (auto &node: graph_ptr->nodes()) {
            if (node->is_fuse()) {
                continue;
            }
            // tff::log::Logger::info("[MemOpt] Optimizing node: {%s} layer id: %d", node->name().c_str(),
            //                        node->layer_id());
            if (auto mem_opt_node = mem_opt_strategy->mem_optimize(node); mem_opt_node != nullptr) {
                graph_ptr->add_node(mem_opt_node);
            }
        }
    }

    //
    void GraphOptimizer::fuse(std::shared_ptr<core::graph::Graph> &graph_ptr) {
        if (graph_ptr == nullptr) {
            tff::log::Logger::error("GraphOptimizer::fuse: graph_ptr is nullptr");
            return;
        }
        for (auto &node: graph_ptr->nodes()) {
            if (node->is_fuse()) {
                continue;
            }
            if (!fuse(graph_ptr, node)) {
                //tff::log::Logger::info("fuse op(%s) do not need fuse!!\n", node->name().c_str());
            }
        }
    }

    bool GraphOptimizer::can_fuse(const std::shared_ptr<graph::GraphNode> &current_node,
                                  std::shared_ptr<graph::GraphNode> &pre_node) const {
        if (current_node->get_tensor()->get_allocator() == nullptr ||
            pre_node->get_tensor()->get_allocator() == nullptr) {
            tff::log::Logger::info("fuse op(%s) allocator is invalid!!\n", current_node->name().c_str());
            return false;
        }
        auto current_device_id = current_node->get_tensor()->get_allocator()->_device_id;
        auto pre_device_id = pre_node->get_tensor()->get_allocator()->_device_id;
        if (current_device_id != pre_device_id) {
            return false;
        }
        const auto match_result = TFF_OP_FUSE_MODEL.find(current_node->op_type());
        if (match_result == TFF_OP_FUSE_MODEL.end()) {
            return false;
        }
        bool bRet = false;
        if (const auto iter =
                    std::ranges::find(match_result->second, pre_node->op_type());
            iter != match_result->second.end()) {
            bRet = true;
        } else {
            bRet = false;
        }

        return bRet;
    }

    bool GraphOptimizer::fuse(std::shared_ptr<core::graph::Graph> &graph_ptr,
                              const std::shared_ptr<graph::GraphNode> &current_node) {
        if (!current_node || current_node->is_fuse()) {
            return false;
        }
        bool bRet = true;
        // // 收集所有可融合的前驱或后驱（必须满足：1. 唯一后继是 current_node；3. 非特殊节点 4. 符合特定融合模式）
        if (current_node->op_type() == graph::TffOpType::TFF_OP_RMS_NORM) {
            std::vector<std::shared_ptr<graph::GraphNode> > succ_fusible_preds; //待融合的算子集合
            //
            for (auto &pred: current_node->output_nodes()) {
                if (!pred || pred->op_type() == graph::TffOpType::TFF_OP_MEM_REF) {
                    continue;
                }

                if (can_fuse(current_node, pred)) {
                    succ_fusible_preds.push_back(pred);
                }
            }
            if (succ_fusible_preds.empty()) {
                return false;
            }
            if (succ_fusible_preds.size() == 1) {
                auto &to_fuse_node = succ_fusible_preds[0];
                if (to_fuse_node->op_type() == graph::TffOpType::TFF_OP_MUL) {
                    to_fuse_node->fuse(current_node);
                    current_node->remove_output_node(to_fuse_node);
                    for (const auto &output_graph_node: to_fuse_node->output_nodes()) {
                        if (output_graph_node->is_fuse()) {
                            continue;
                        }
                        current_node->add_output_node(output_graph_node);
                        output_graph_node->add_input_node(current_node);
                        output_graph_node->remove_input_node(to_fuse_node);
                    }
                    //graph_ptr->add_node(fuse_node);
                } else {
                    tff::log::Logger::error("fuse op(%s) fuse error\n", current_node->name().c_str());
                    return false;
                }
            } else if (succ_fusible_preds.size() == 2) {
                //todo add bias;
            }
        }
        else if (current_node->op_type() == graph::TffOpType::TFF_OP_FLASH_ATTN_EXT ||
                   current_node->op_type() == graph::TffOpType::TFF_OP_FLASH_ATTN_PAGED) {
            std::vector<std::shared_ptr<graph::GraphNode> > pre_fusible_preds; //待融合的算子集合
            for (auto &pred: current_node->input_nodes()) {
                if (!pred || pred->op_type() == graph::TffOpType::TFF_OP_MEM_REF) {
                    continue;
                }

                if (can_fuse(current_node, pred)) {
                    pre_fusible_preds.push_back(pred);
                }
            }

            if (pre_fusible_preds.empty()) {
                return false;
            }
            for (const auto &to_fuse_node: pre_fusible_preds) {
                to_fuse_node->fuse(current_node, false);
                for (const auto &input_graph_node: to_fuse_node->input_nodes()) {
                    if (input_graph_node->is_fuse()) {
                        continue;
                    }

                    replace_input_node(current_node, input_graph_node, to_fuse_node);
                }
            }
        }
        bRet = fuse_same_node(current_node);//融合重复节点;
        return bRet;
    }

    bool GraphOptimizer::fuse_quant_node(const std::shared_ptr<graph::GraphNode> &current_node) const {
        for (auto &pred: current_node->input_nodes()) {
            if (pred->output_nodes().size() == 1) {
                continue;
            }
            for (auto &pred_op_node: pred->output_nodes()) {
                if (pred_op_node->is_fuse()) {
                    continue;
                }
                if (pred_op_node == current_node) {
                    continue;
                }
                if (pred_op_node->op_type() == graph::TffOpType::TFF_OP_QUANTIZE) {
                    pred_op_node->fuse();
                    pred_op_node->remove_input_node(pred);
                    pred->remove_output_node(pred_op_node);
                    for (auto &next_op_node: pred_op_node->output_nodes()) {
                        if (next_op_node->is_fuse()) {
                            continue;
                        }
                        const auto &builder =
                                std::dynamic_pointer_cast<tff::kernel::QuantBuilder>(current_node->builder());
                        const auto &next_builder =
                                std::dynamic_pointer_cast<tff::kernel::QuantMatMulBuilder>(next_op_node->builder());
                        if (next_builder) {
                            next_builder->x(builder->out<std::shared_ptr<core::memory::Tensor> >());
                        } else {
                            tff::log::Logger::error("fuse op(%s) fuse error\n", current_node->name().c_str());
                        }
                        next_op_node->remove_input_node(pred_op_node);
                        next_op_node->add_input_node(current_node);
                        pred_op_node->remove_output_node(next_op_node);
                        current_node->add_output_node(next_op_node);
                    }
                }
            }
        }
        return true;
    }

    bool GraphOptimizer::fuse_same_node(const std::shared_ptr<graph::GraphNode> &current_node) const {
        if (!is_need_fuse_same_node(current_node)) {
            return false;
        }
        std::unordered_map<graph::GraphNodeSemanticKey, std::shared_ptr<graph::GraphNode>> seen;
        bool bRet = false;
        for (const auto& node : current_node->output_nodes()) {
            graph::GraphNodeSemanticKey key(node);
            if (seen.contains(key)) {
                auto& remain_node = seen[key];
                auto& same_node = node;
                fuse_same_node(current_node, remain_node, same_node);
                bRet = true;
            } else {
                seen[key] = node;
            }
        }
        return bRet;
    }

    bool GraphOptimizer::is_need_fuse_same_node(const std::shared_ptr<graph::GraphNode> &current_node) const {
        std::unordered_set<core::graph::GraphNodeSemanticKey> seen;

        for (const auto &node : current_node->output_nodes()) {
            if (node->is_fuse()) {
                continue;
            }
            graph::GraphNodeSemanticKey key(node);
            if (!seen.insert(key).second) {
                // tff::log::Logger::info("Duplicate node detected for fusion: %s (Op: %d)",
                //     node->name().c_str(), (int)node->op_type());
                return true;
            }
        }
        return false;
    }

    void GraphOptimizer::fuse_same_node(const std::shared_ptr<graph::GraphNode> &current_node,
                                        const std::shared_ptr<graph::GraphNode> &remain_node,
                                        const std::shared_ptr<graph::GraphNode> &same_node) const {
        same_node->fuse();
        same_node->remove_input_node(current_node);
        current_node->remove_output_node(same_node);
        for (auto &next_op_node: same_node->output_nodes()) {
            if (next_op_node->is_fuse()) {
                continue;
            }
            auto param_name = next_op_node->para_name(same_node->get_tensor());
            if (param_name != nullptr) {
                const auto &dst_builder = next_op_node->builder();
                dst_builder->build()->set_param(param_name, remain_node->get_tensor());
            }
            next_op_node->remove_input_node(same_node);
            next_op_node->add_input_node(remain_node);
            same_node->remove_output_node(next_op_node);
            remain_node->add_output_node(next_op_node);
        }
    }

    void GraphOptimizer::insert_input_node(const std::shared_ptr<graph::GraphNode> &current_node,
        const std::shared_ptr<graph::GraphNode> &new_input_node,
        const std::shared_ptr<graph::GraphNode> &old_input_node) const {
        current_node->remove_input_node(old_input_node);
        current_node->add_input_node(new_input_node);
        new_input_node->add_output_node(current_node);

        new_input_node->add_input_node(old_input_node);
        old_input_node->add_output_node(new_input_node);
        auto param_name = current_node->para_name(old_input_node->get_tensor());
        if (param_name != nullptr) {
            const auto &dst_builder = current_node->builder();
            dst_builder->build()->set_param(param_name, new_input_node->get_tensor());
        }
        current_node->shape_infer();
    }
    //
    void GraphOptimizer::replace_input_node(const std::shared_ptr<graph::GraphNode> &current_node,
                        const std::shared_ptr<graph::GraphNode> &new_input_node,
                        const std::shared_ptr<graph::GraphNode> &old_input_node) const {
        current_node->remove_input_node(old_input_node);
        old_input_node->remove_output_node(current_node);

        current_node->add_input_node(new_input_node);
        new_input_node->add_output_node(current_node);

        auto param_name = old_input_node->para_name(new_input_node->get_tensor());
        if (!new_input_node->is_leaf()) {
            param_name = current_node->para_name(old_input_node->get_tensor());
        }
        if (param_name != nullptr) {
            const auto &dst_builder = current_node->builder();
            dst_builder->build()->set_param(param_name, new_input_node->get_tensor());
        }
    }
}
