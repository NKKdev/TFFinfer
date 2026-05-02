//
// Created by nkk on 2026/2/19.
//
#include <memory>
#include "ModelCreatorBase.h"

#include "device/DeviceManager.h"
#include "graph/GraphNode.h"
#include "include/TFFOPCreator.h"

namespace tff::core::model {
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_ffn_up(
        std::shared_ptr<tff::core::graph::GraphNode> &weight_node,
        std::shared_ptr<tff::core::graph::GraphNode> &x_node) {
        if (weight_node == nullptr || x_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_ffn_up] weight_node or x_node is null");
            return nullptr;
        }
        auto ffn_up_node = build_mul_mat_node(
            weight_node, x_node);
        return ffn_up_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_ffn_gate(
        std::shared_ptr<tff::core::graph::GraphNode> &weight_node,
        std::shared_ptr<tff::core::graph::GraphNode> &x_node) {
        if (weight_node == nullptr || x_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_ffn_up] weight_node or x_node is null");
            return nullptr;
        }
        auto ffn_gate = build_mul_mat_node(
            weight_node, x_node);
        return ffn_gate;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_ffn_down(
        std::shared_ptr<tff::core::graph::GraphNode> &weight_node,
        std::shared_ptr<tff::core::graph::GraphNode> &x_node) {
        if (weight_node == nullptr || x_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_ffn_up] weight_node or x_node is null");
            return nullptr;
        }
        auto ffn_down_node = build_mul_mat_node(
            weight_node, x_node);
        return ffn_down_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_ffn(
        tff::core::graph::TFFUnaryType type,
        const int &layer_id,
        std::shared_ptr<tff::core::graph::GraphNode> &x_node) {
        if (x_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_ffn] x_node is null");
            return nullptr;
        }
        auto ffn_up_node = build_ffn_up(this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING]
                                        [layer_id][memory::ModelTensorType::LLM_TENSOR_FFN_UP], x_node);

        auto ffn_gate_node = build_ffn_gate(this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING]
                                            [layer_id][memory::ModelTensorType::LLM_TENSOR_FFN_GATE], x_node);

        //silu;
        auto unary_op_node = build_unary_op(type, ffn_up_node, ffn_gate_node);

        return build_ffn_down(this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING]
                              [layer_id][memory::ModelTensorType::LLM_TENSOR_FFN_DOWN], unary_op_node);
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_unary_op(
        tff::core::graph::TFFUnaryType type,
        std::shared_ptr<tff::core::graph::GraphNode> &up_node,
        std::shared_ptr<tff::core::graph::GraphNode> &gate_node) {
        if (up_node == nullptr || gate_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_unary_op] up_node or gate_node is null");
            return nullptr;
        }
        auto unary_op_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_UNARY);
        unary_op_node->set_node_meta(graph::NodeMetadata{gate_node->name() + "_swiglu_unary_op"});

        const auto &builder =
                std::dynamic_pointer_cast<tff::kernel::UnaryOPBuilder>(unary_op_node->builder());
        builder->unary_type(static_cast<int>(type))
                .x1(gate_node->get_tensor()).x2(up_node->get_tensor());
        unary_op_node->shape_infer();

        unary_op_node->add_input_node(gate_node);
        unary_op_node->add_input_node(up_node);

        return unary_op_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_mul_node(
        std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &weight_node) {
        if (a_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_mul_node] a_node or b_node is null");
            return nullptr;
        }
        if (weight_node == nullptr) {
            return a_node;
        }
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MUL);
        out_put_node->set_node_meta(graph::NodeMetadata({a_node->name() + "_mul_w"}));

        const auto &builder = std::dynamic_pointer_cast<kernel::MulBuilder>(out_put_node->builder());
        builder->weight(weight_node->get_tensor())
                .x(a_node->get_tensor());
        out_put_node->shape_infer();
        out_put_node->add_input_node(a_node);
        out_put_node->add_input_node(weight_node);
        return out_put_node;
    }

    //
    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_mul_mat_node(
        std::shared_ptr<tff::core::graph::GraphNode> &weight_node,
        std::shared_ptr<tff::core::graph::GraphNode> &x_node) {
        if (weight_node == nullptr || x_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_mul_mat_node] weight_node or x_node is null");
            return nullptr;
        }
        tff::core::graph::TffOpType op_type;
        auto weight_data_type = weight_node->get_tensor()->get_data_type();
        switch (weight_data_type) {
            case tff::core::memory::DataType::TFF_DATA_TYPE_Q8_0:
            case memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED:
                op_type = tff::core::graph::TffOpType::TFF_OP_QUANTIZE_MATMUL;
                break;
            case tff::core::memory::DataType::TFF_DATA_TYPE_F32:
            case tff::core::memory::DataType::TFF_DATA_TYPE_F64:
            case tff::core::memory::DataType::TFF_DATA_TYPE_F16:
            default:
                op_type = tff::core::graph::TffOpType::TFF_OP_MUL_MAT;
                break;
        }

        auto result = ADD_NODE(op_type);
        result->set_node_meta(graph::NodeMetadata{weight_node->name() + "_mul_w"});

        const auto &builder =
                std::dynamic_pointer_cast<kernel::QuantMatMulBuilder>(result->builder());
        auto &input_tensor = weight_node->get_tensor();

        if ((weight_data_type == tff::core::memory::DataType::TFF_DATA_TYPE_Q8_0 ||
             weight_data_type == tff::core::memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED)) {
            auto quantize_node = ADD_NODE(graph::TFF_OP_QUANTIZE);
            std::string quantize_node_name = weight_node->name() +
                                             ".quantize_q_8_0_node";
            quantize_node->set_node_meta(graph::NodeMetadata{quantize_node_name});

            const auto &quant_builder =
                    std::dynamic_pointer_cast<kernel::QuantBuilder>(quantize_node->builder());
            quant_builder->quant_data_type(weight_data_type).in(x_node->get_tensor());

            quantize_node->shape_infer();
            quantize_node->add_input_node(x_node);
            builder->weight(input_tensor).x(quantize_node->get_tensor());

            result->shape_infer();
            result->add_input_node(quantize_node);
        } else {
            builder->weight(input_tensor).x(x_node->get_tensor());
            result->shape_infer();
            result->add_input_node(x_node);
        }

        result->add_input_node(weight_node);
        return result;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_add_node(
        std::shared_ptr<tff::core::graph::GraphNode> &a_node,
        std::shared_ptr<tff::core::graph::GraphNode> &b_node) {
        if (a_node == nullptr || b_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_add_node] a_node or b_node is null");
            return nullptr;
        }
        auto out_put_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_BINARY);
        out_put_node->set_node_meta(tff::core::graph::NodeMetadata(b_node->name() + "_add_node"));
        const auto &builder =
                std::dynamic_pointer_cast<tff::kernel::BinaryOPBuilder>(out_put_node->builder());
        builder->binary_type(static_cast<int>(core::graph::TFFBinaryType::TFF_BINARY_TYPE_ADD))
                .x1(a_node->get_tensor()).x2(b_node->get_tensor());
        out_put_node->shape_infer();
        out_put_node->add_input_node(a_node);
        out_put_node->add_input_node(b_node);
        return out_put_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_norm(
        graph::TFFNormType type,
        std::shared_ptr<tff::core::graph::GraphNode> &weight_node,
        std::shared_ptr<tff::core::graph::GraphNode> &x_node) {
        if (x_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_norm] weight_node or x_node is null");
            return nullptr;
        }
        auto rms_norm_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
        rms_norm_node->set_node_meta(graph::NodeMetadata{weight_node->name() + "_rms_norm"});

        const auto &builder =
                std::dynamic_pointer_cast<tff::kernel::NormBuilder>(rms_norm_node->builder());
        builder->epsilon(this->_graph_ctx._f_norm_rms_eps)
                .in(x_node->get_tensor())
                .norm_type(type);
        rms_norm_node->shape_infer();

        rms_norm_node->add_input_node(x_node);

        auto rms_norm_mul_w_node = build_mul_node(rms_norm_node,
                                                  weight_node);
        return rms_norm_mul_w_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_reshape_node(
        std::shared_ptr<tff::core::graph::GraphNode> &input_node,
        int dim0, int dim1, int dim2, int dim3) {
        if (input_node == nullptr) {
            tff::log::Logger::error("input node is null");
            return nullptr;
        }
        auto reshape_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_RESHAPE);
        reshape_node->set_node_meta(graph::NodeMetadata{input_node->name() + "_reshape"});

        const auto &builder =
                std::dynamic_pointer_cast<tff::kernel::ReshapeBuilder>(reshape_node->builder());
        builder->embd_head_num(dim0).head_num(dim1).token_num(dim2).batch_num(dim3).in(input_node->get_tensor());
        reshape_node->shape_infer();
        reshape_node->add_input_node(input_node);

        return reshape_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_attn(
        const char *name,
        const int &layer_id,
        std::shared_ptr<tff::core::graph::GraphNode> &q_node,
        std::shared_ptr<tff::core::graph::GraphNode> &k_node,
        std::shared_ptr<tff::core::graph::GraphNode> &v_node) {
        if (q_node == nullptr || k_node == nullptr || v_node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_attn] q_node, k_node, v_node can not be nullptr");
            return nullptr;
        }
        auto flash_attn_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_FLASH_ATTN_PAGED);
        graph::NodeMetadata meta{std::string(name) + "flash_attn"};
        flash_attn_node->set_node_meta(meta);
        const auto &builder =
                std::dynamic_pointer_cast<kernel::PagedFlashAttnBuilder>(flash_attn_node->builder());
        builder->q(q_node->get_tensor())
                .k(k_node->get_tensor())
                .v(v_node->get_tensor());
        if (this->_graph_ctx._mask == nullptr) {
            auto mask_node = build_mask_node();
            builder->mask(this->_graph_ctx._mask);
            flash_attn_node->shape_infer();
            flash_attn_node->add_input_node(mask_node);
        } else {
            builder->mask(this->_graph_ctx._mask);
            flash_attn_node->shape_infer();
        }
        flash_attn_node->add_input_node(q_node);
        flash_attn_node->add_input_node(k_node);
        flash_attn_node->add_input_node(v_node);
        auto flash_attn_reshape_node = build_reshape_node(flash_attn_node,
                                                          this->_graph_ctx._n_embd_head * this->_graph_ctx._n_head,
                                                          this->_graph_ctx._n_tokens,
                                                          1,
                                                          1);

        auto flash_attn_mul_w_node = build_mul_mat_node(
            this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][memory::ModelTensorType::LLM_TENSOR_ATTN_OUT],
            flash_attn_reshape_node);

        return flash_attn_mul_w_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_kv_cache_store_node(
        tff::core::memory::ModelTensorType tensor_type,
        const int &layer_id,
        const std::shared_ptr<graph::GraphNode> &kv_node) {
        if (kv_node == nullptr) {
            tff::log::Logger::error("kv_node is null");
            return nullptr;
        }
        auto cache_store_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_SET_ROWS);
        if (tensor_type == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
            std::string cache_store_node_name = "blk." + std::to_string(layer_id) + ".k_cache_store_node";
            cache_store_node->set_node_meta(graph::NodeMetadata{cache_store_node_name});
        } else if (tensor_type == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
            std::string cache_store_node_name = "blk." + std::to_string(layer_id) + ".v_cache_store_node";
            cache_store_node->set_node_meta(graph::NodeMetadata{cache_store_node_name});
        }
        if (kv_node->get_tensor() == nullptr || kv_node->get_tensor()->get_allocator() == nullptr) {
            tff::log::Logger::error("kv_node allocator is null");
            return nullptr;
        }
        auto device_id = kv_node->get_tensor()->get_allocator()->_device_id;
        if (this->_graph_ctx._pre_token_num == 0) {
            this->_graph_ctx._kv_cache_ptr[device_id]->set_kv(this->_graph_ctx._seq_id,
                                                         layer_id,
                                                         this->_graph_ctx._n_tokens);
            this->_graph_ctx._pre_token_num =
                this->_graph_ctx._kv_cache_ptr[device_id]->get_kv_token_num(this->_graph_ctx._seq_id, layer_id);
        }else {
            this->_graph_ctx._pre_token_num = 0;
        }
        auto data_type = memory::DataType::TFF_DATA_TYPE_F32;
        if (this->_graph_ctx._use_fp16) {
            data_type = memory::DataType::TFF_DATA_TYPE_F16;
        }

        const auto &builder =
                std::dynamic_pointer_cast<kernel::SetRowBuilder>(cache_store_node->builder());
        builder->seq_id(this->_graph_ctx._seq_id)
                .layer_id(layer_id)
                .in(kv_node->get_tensor())
                .kv_cache_ctx(this->_graph_ctx._kv_cache_ptr[device_id])
                .data_type(data_type).tensor_type(tensor_type);
        cache_store_node->shape_infer();

        cache_store_node->add_input_node(kv_node);
        return cache_store_node;
    }

    std::shared_ptr<graph::GraphNode> ModelCreatorBase::build_mask_node() {
        auto data_type = this->_graph_ctx._use_fp16
                             ? memory::DataType::TFF_DATA_TYPE_F16
                             : memory::DataType::TFF_DATA_TYPE_F32;
        std::array<int64_t, MAX_TENSOR_DIM> shapes = {this->_graph_ctx._max_seq_len, this->_graph_ctx._max_seq_len, 1, 1};
        this->_graph_ctx._mask = std::make_shared<memory::Tensor>(data_type,
                                                                  memory::MemoryType::TFF_MEM_TYPE_RESIDENT, shapes);
        auto device = tff::factory::ModuleFactory::instance()->create_shared<
            tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
        std::vector<int> device_ids;
        device->get_device_id(device_ids);
        this->_graph_ctx._mask->set_allocator(device->get_device_buffer_allocator(device_ids[0]));


        auto attn_mask_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ATTN_MASK);
        graph::NodeMetadata meta_attn_mask{true, false, "input_attn_mask"};
        attn_mask_node->set_node_meta(meta_attn_mask);
        const auto &mask_builder =
                std::dynamic_pointer_cast<tff::kernel::MaskOPBuilder>(attn_mask_node->builder());
        mask_builder->mask_type(static_cast<const int>(graph::TFF_MASK_TYPE_CAUSAL))
                .token_num(this->_graph_ctx._max_seq_len)
                .data_type(data_type).in(this->_graph_ctx._mask);
        attn_mask_node->shape_infer();
        return attn_mask_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_kv_cache_load_node(
        const tff::core::memory::ModelTensorType tensor_type,
        const int &layer_id,
        const std::shared_ptr<graph::GraphNode> &node) {
        if (node == nullptr) {
            tff::log::Logger::error("[ModelCreatorBase::build_kv_cache_load_node] node is null");
            return nullptr;
        }
        auto cache_load_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_GET_ROWS);
        if (tensor_type == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
            std::string cache_load_node_name = "blk." + std::to_string(layer_id) + ".k_cache_load_node";
            cache_load_node->set_node_meta(graph::NodeMetadata{cache_load_node_name});
        } else if (tensor_type == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
            std::string cache_load_node_name = "blk." + std::to_string(layer_id) + ".v_cache_load_node";
            cache_load_node->set_node_meta(graph::NodeMetadata{cache_load_node_name});
        }

        if (node == nullptr || node->get_tensor() == nullptr || node->get_tensor()->get_allocator() == nullptr) {
            tff::log::Logger::error("kv_node allocator is null");
            return nullptr;
        }
        auto &input_tensor = node->get_tensor();
        auto device_id = input_tensor->get_allocator()->_device_id;

        const auto &builder =
                std::dynamic_pointer_cast<kernel::GetRowBuilder>(cache_load_node->builder());
        builder->seq_id(this->_graph_ctx._seq_id)
                .layer_id(layer_id)
                .kv_cache_ctx(this->_graph_ctx._kv_cache_ptr[device_id]).in(input_tensor)
        .max_seq_len(this->_graph_ctx._max_seq_len);
        cache_load_node->shape_infer();

        cache_load_node->add_input_node(node);

        return cache_load_node;
    }

    std::shared_ptr<graph::GraphNode> ModelCreatorBase::
    build_convert_node(memory::DataType type, std::shared_ptr<tff::core::graph::GraphNode> &node) {
        auto convert_fp16_node = ADD_NODE(graph::TFF_OP_CONVERT);
        convert_fp16_node->set_node_meta(graph::NodeMetadata{node->name() + "_convert_fp16_node"});

        const auto &builder =
                std::dynamic_pointer_cast<kernel::ConvertOPBuilder>(convert_fp16_node->builder());
        builder->in(node->get_tensor()).convert_data_type(type);
        convert_fp16_node->shape_infer();
        convert_fp16_node->add_input_node(node);
        return convert_fp16_node;
    }

    //
    void ModelCreatorBase::build_weight_node(
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t, std::unordered_map<tff::
            core::memory::ModelTensorType, std::shared_ptr<tff::core::model::layer::ModelLayerObject> > > > &
        layer_map) {
        for (const auto &layer_type_iter: layer_map) {
            auto &weight_layer_map = this->_weight_node_map[layer_type_iter.first];
            for (const auto &layer_iter: layer_type_iter.second) {
                auto &layer_node_map = weight_layer_map[layer_iter.first];
                for (const auto &tensor_type_iter: layer_iter.second) {
                    auto weight_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_REF);
                    weight_node->set_node_meta(graph::NodeMetadata{
                        tensor_type_iter.second->_layer_name
                    });
                    weight_node->set_layer_id(layer_iter.first);
                    const auto &builder =
                            std::dynamic_pointer_cast<kernel::MemRefBuilder>(weight_node->builder());
                    builder->in(tensor_type_iter.second->_tensor).out(tensor_type_iter.second->_tensor);
                    weight_node->shape_infer();
                    layer_node_map[tensor_type_iter.first] = weight_node;
                }
            }
        }
    }

    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_host_node(
        std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node) {
        NodeType out_put_node;
        auto current_map2cpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
        tff::core::graph::NodeMetadata meta_map2cpu{layer->_layer_name + "_map2cpu"};
        current_map2cpu_node->set_node_meta(meta_map2cpu);
        auto device = tff::factory::ModuleFactory::instance()->create_shared<
            tff::core::device::DeviceBaseObject>(
            DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
        std::vector<int> device_ids;
        device->get_device_id(device_ids);
        std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > devices = {
            {device_ids[0], device}
        };
        current_map2cpu_node->bind_devices(devices);

        const auto &builder =
                std::dynamic_pointer_cast<kernel::Map2CpuBuilder>(current_map2cpu_node->builder());
        builder->file_idx(layer->_model_file_index)
                .offset(layer->_offset).size(layer->_data_size).model_ctx(this->_graph_ctx._model_loader)
                .in(layer->_tensor);
        current_map2cpu_node->shape_infer();
        current_map2cpu_node->add_input_node(input_node[tff::core::graph::GraphNodeType::TFF_GRAPH_NODE_MAP2CPU]);

        return current_map2cpu_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_device_node(
        std::shared_ptr<layer::ModelLayerObject> &layer, NodeType &input_node,
        std::shared_ptr<tff::core::graph::GraphNode> &current_cpu_node, bool is_input) {
        auto current_cpu2gpu_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
        current_cpu2gpu_node->set_node_meta(graph::NodeMetadata{layer->_layer_name + "_cpu2gpu"});
        auto device_id = 0;
        if (!is_input) {
            current_cpu2gpu_node->bind_devices(layer->_device_list);
            device_id = layer->_device_list.begin()->first;
        } else {
            auto device = tff::factory::ModuleFactory::instance()->create_shared<
                tff::core::device::DeviceBaseObject>(
                DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
            std::vector<int> device_ids;
            device->get_device_id(device_ids);
            std::unordered_map<int, std::shared_ptr<tff::core::device::DeviceBaseObject> > devices = {
                {device_ids[0], device}
            };
            current_cpu2gpu_node->bind_devices(devices);
            device_id = device_ids[0];
        }
        auto src_type = device::DeviceType::TFF_BACKEND_DEVICE_TYPE_CPU;
        auto dst_type = device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU;

        const auto &builder =
                std::dynamic_pointer_cast<kernel::MemCpyBuilder>(current_cpu2gpu_node->builder());
        builder->memcpy_kind((make_cpy_kind(src_type, dst_type)))
                .in(current_cpu_node->get_tensor()).out(layer->_tensor)
                .source_id(-1).dest_id(device_id);
        current_cpu2gpu_node->shape_infer();


        auto iter = input_node.find(graph::GraphNodeType::TFF_GRAPH_NODE_CPU2GPU);
        if (iter != input_node.end()) {
            current_cpu2gpu_node->add_input_node(iter->second);
        }
        current_cpu2gpu_node->add_input_node(current_cpu_node);
        return current_cpu2gpu_node;
    }

    NodeType ModelCreatorBase::build_layer_node(memory::ModelTensorType tensor_type,
                                                const std::unordered_map<tff::core::memory::ModelTensorType,
                                                    std::shared_ptr
                                                    <
                                                        tff::core::model::layer::ModelLayerObject> > &layer_map,
                                                NodeType &input_node, bool
                                                is_input) {
        auto layer = layer_map.find(tensor_type)->second;
        if (layer == nullptr) {
            return NodeType();
        }
        NodeType out_put_node;
        out_put_node[graph::TFF_GRAPH_NODE_MAP2CPU] = build_host_node(layer, input_node);
        if (!is_input) {
            auto gpu_node = build_device_node(layer, input_node,
                                              out_put_node[graph::TFF_GRAPH_NODE_MAP2CPU], is_input);
            if (layer->_tensor->get_data_type() == core::memory::DataType::TFF_DATA_TYPE_Q8_0) {
                auto aligned_node = build_aligned_node(gpu_node);

                out_put_node[graph::TFF_GRAPH_NODE_CPU2GPU] = aligned_node;
            } else {
                out_put_node[graph::TFF_GRAPH_NODE_CPU2GPU] = gpu_node;
            }
        } else {
            out_put_node[graph::TFF_GRAPH_NODE_MAP2CPU]->get_tensor()->set_memory_type(
                core::memory::MemoryType::TFF_MEM_TYPE_RESIDENT);
            out_put_node[graph::TFF_GRAPH_NODE_CPU2GPU] = out_put_node[graph::TFF_GRAPH_NODE_MAP2CPU];
        }

        auto mem_ref_node = ADD_NODE(core::graph::TffOpType::TFF_OP_MEM_REF);
        mem_ref_node->set_node_meta(graph::NodeMetadata{
            out_put_node[graph::TFF_GRAPH_NODE_CPU2GPU]->name() + "_ref_node"
        });
        const auto &builder = std::dynamic_pointer_cast<kernel::MemRefBuilder>(mem_ref_node->builder());
        builder->in(out_put_node[graph::TFF_GRAPH_NODE_CPU2GPU]->get_tensor()).out(layer->_tensor);
        mem_ref_node->shape_infer();
        mem_ref_node->add_input_node(out_put_node[graph::TFF_GRAPH_NODE_CPU2GPU]);
        out_put_node[graph::TFF_GRAPH_NODE_CPU2GPU] = mem_ref_node;
        return out_put_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> ModelCreatorBase::build_aligned_node(
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        tff::core::graph::TffOpType alinged_op_type;
        core::memory::DataType alinged_data_type;
        switch (input_node->get_tensor()->get_data_type()) {
            case core::memory::DataType::TFF_DATA_TYPE_Q8_0:
                alinged_op_type = tff::core::graph::TffOpType::TFF_OP_QUANTIZE_ALIGNED;
                alinged_data_type = core::memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED;
                break;
            default:
                break;
        }
        auto aligned_node = ADD_NODE(alinged_op_type);
        aligned_node->set_node_meta(graph::NodeMetadata{input_node->name() + "_aligned_node"});
        const auto &builder =
                std::dynamic_pointer_cast<kernel::QuantAlignedBuilder>(aligned_node->builder());
        builder->quant_data_type(alinged_data_type).in(input_node->get_tensor());
        aligned_node->shape_infer();
        aligned_node->add_input_node(input_node);


        return aligned_node;
    }

    std::shared_ptr<graph::GraphNode> ModelCreatorBase::build_mem_cpy_node(const int &source_device_id,
                                                                           const int &dest_device_id,
                                                                           std::shared_ptr<graph::GraphNode> &node) {
        auto mem_cpy_node = ADD_NODE(graph::TFF_OP_MEM_CPY);
        mem_cpy_node->set_node_meta(graph::NodeMetadata(node->name() + "_mem_cpy"));
        auto device_manager = std::dynamic_pointer_cast<device::DeviceManager>(
            tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                DEVICE_MANAGER_FLAG,
                tff::factory::ModuleKeyType(DEVICE_MANAGER_FLAG)));
        auto src_type = device_manager->get_device(source_device_id)->device_type();
        auto dst_type = device_manager->get_device(dest_device_id)->device_type();
        auto builder = std::make_shared<kernel::MemCpyBuilder>();
        builder->memcpy_kind(make_cpy_kind(src_type, dst_type)).in(node->get_tensor())
                .source_id(source_device_id).dest_id(dest_device_id);
        mem_cpy_node->set_builder(builder);
        mem_cpy_node->shape_infer();

        mem_cpy_node->add_input_node(node);
        return mem_cpy_node;
    }
}
