//
// Created by nkk on 2025/11/10.
//

#include "QWen3Creator.h"

#include "device/DeviceManager.h"
#include "include/TFFOPCreator.h"

namespace tff::core::model {
    REGISTER_MODULE_OBJECT(QWen3Creator, ModelCreatorBase, MODEL_CREATOR_FLAG,
                           tff::core::model::to_string(tff::core::model::ModelArchitectureType::TFF_MODEL_ARCH_QWEN3));

    //
    void QWen3Creator::build_graph(
        std::unordered_map<tff::core::model::ModelTensorLayerType, std::unordered_map<uint32_t,
            std::unordered_map<tff::core::memory::ModelTensorType, std::shared_ptr<
                tff::core::model::layer::ModelLayerObject> > > > &layer_map,
        std::shared_ptr<tff::core::graph::Graph> &graph_ptr) {
        //
        if (!graph_ptr) {
            graph_ptr = std::make_shared<tff::core::graph::Graph>();
        }

        const auto input_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_INPUT);
        if (input_layer_iter == layer_map.end()) {
            return;
        }
        const auto repeating_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_REPEATING);
        if (repeating_layer_iter == layer_map.end()) {
            return;
        }
        //
        const auto output_layer_iter = layer_map.find(
            tff::core::model::ModelTensorLayerType::LLM_TENSOR_LAYER_OUTPUT);
        if (output_layer_iter == layer_map.end()) {
            return;
        }
        build_weight_node(layer_map);

        auto input_node = build_inputs(input_layer_iter->second.begin()->second);

        //
        if (input_layer_iter != layer_map.end() && !input_layer_iter->second.empty() &&
            repeating_layer_iter != layer_map.end() && !repeating_layer_iter->second.empty()) {
            //
            if (repeating_layer_iter != layer_map.end() && repeating_layer_iter->second.size() > 1) {
                const auto &repeating_layer_map = repeating_layer_iter->second;
                const int n_layers = repeating_layer_map.size();
                for (int layer_id = 0; layer_id < n_layers; ++layer_id) {
#ifdef _DEBUG1
                    {
                        if (layer_id > 20) {
                            continue;
                        }
                    }
#endif
                    //tff::log::Logger::info("build layer :%d graph\n", layer_id);
                    input_node->set_layer_id(layer_id);
                    auto attn_norm_node = build_norm(this->_graph_ctx._norm_type,
                                                     this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][
                                                         memory::ModelTensorType::LLM_TENSOR_ATTN_NORM],
                                                     input_node);
                    auto attn_q_node = build_qkv_node(this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][
                                                          memory::ModelTensorType::LLM_TENSOR_ATTN_Q],
                                                      attn_norm_node);

                    auto attn_k_node = build_qkv_node(this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][
                                                          memory::ModelTensorType::LLM_TENSOR_ATTN_K],
                                                      attn_norm_node);
                    auto attn_v_node = build_qkv_node(this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][
                                                          memory::ModelTensorType::LLM_TENSOR_ATTN_V],
                                                      attn_norm_node);

                    auto attn_q_reshape_node = build_reshape_node(attn_q_node,
                                                                  this->_graph_ctx._n_embd_head,
                                                                  this->_graph_ctx._n_head,
                                                                  this->_graph_ctx._n_tokens,
                                                                  1);
                    auto attn_k_reshape_node = build_reshape_node(attn_k_node,
                                                                  this->_graph_ctx._n_embd_head,
                                                                  this->_graph_ctx._n_head_kv,
                                                                  this->_graph_ctx._n_tokens,
                                                                  1);
                    auto attn_v_reshape_node = build_reshape_node(attn_v_node,
                                                                  this->_graph_ctx._n_embd_head,
                                                                  this->_graph_ctx._n_head_kv,
                                                                  this->_graph_ctx._n_tokens,
                                                                  1);

                    auto attn_q_norm_node = build_norm(this->_graph_ctx._norm_type,
                                                       this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][
                                                           tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q_NORM],
                                                       attn_q_reshape_node);

                    auto attn_k_norm_node = build_norm(this->_graph_ctx._norm_type,
                                                       this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][
                                                           tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K_NORM],
                                                       attn_k_reshape_node);
                    //

                    auto attn_k_store_cache_node = build_kv_cache_store_node(
                        memory::ModelTensorType::LLM_TENSOR_ATTN_K, layer_id, attn_k_norm_node);

                    auto attn_v_store_cache_node = build_kv_cache_store_node(
                        memory::ModelTensorType::LLM_TENSOR_ATTN_V, layer_id, attn_v_reshape_node);

                    auto attn_k_load_cache_node = build_kv_cache_load_node(
                        memory::ModelTensorType::LLM_TENSOR_ATTN_K,
                        layer_id, attn_k_store_cache_node);

                    auto attn_v_load_cache_node = build_kv_cache_load_node(
                        memory::ModelTensorType::LLM_TENSOR_ATTN_V, layer_id, attn_v_store_cache_node);

                    //

                    auto k_rope_node = build_rope_node(layer_id,
                                                       attn_k_load_cache_node);
                    std::shared_ptr<core::graph::GraphNode> q_rope_node;
                    if (this->_graph_ctx._use_fp16) {
                        auto convert_fp16_node = build_convert_node(memory::DataType::TFF_DATA_TYPE_F16,
                                                                    attn_q_norm_node);
                        q_rope_node = build_rope_node(layer_id, convert_fp16_node);
                    } else {
                        q_rope_node = build_rope_node(layer_id, attn_q_norm_node);
                    }
                    std::string name = "blk." + std::to_string(layer_id) + ".";
                    auto attn_node = build_attn(name.c_str(),
                                                layer_id, q_rope_node, k_rope_node, attn_v_load_cache_node);

                    auto node = attn_node;
                    auto inp_node = input_node;
                    // if (layer_id == n_layers - 1) {
                    //     node = build_gather_node(attn_node);
                    //     inp_node = build_gather_node(input_node);
                    // }
                    //
                    auto ffn_inp_node = build_ffn_inp(inp_node, node);

                    auto ffn_norm_node = build_norm(this->_graph_ctx._norm_type,
                                                    this->_weight_node_map[LLM_TENSOR_LAYER_REPEATING][layer_id][
                                                        memory::ModelTensorType::LLM_TENSOR_FFN_NORM],
                                                    ffn_inp_node);

                    //
                    auto ffn_node = build_ffn(TFF_UNARY_TYPE_SILU, layer_id, ffn_norm_node);

                    auto result_node = build_add_node(ffn_inp_node, ffn_node);
                    input_node = result_node;
                }
            }
        }

        auto output_norm_node = build_norm(this->_graph_ctx._norm_type,
                                           this->_weight_node_map[LLM_TENSOR_LAYER_OUTPUT]
                                           [0][memory::ModelTensorType::LLM_TENSOR_OUTPUT_NORM],
                                           input_node);
        output_norm_node->set_node_meta(NodeMetadata{false, true, "output_norm_node"});
        auto out_put_node = build_output(memory::ModelTensorType::LLM_TENSOR_OUTPUT,
                                         output_norm_node);
        out_put_node->set_node_meta(NodeMetadata{false, true, "output_node"});
        auto offload_node = build_offload_node(out_put_node);
        graph_ptr->build_graph(offload_node);
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_rope_table_node() {
        auto rope_table_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_PRE_ROPE_TABLE);
        if (rope_table_node != nullptr) {
            NodeMetadata meta_rope_node{true, false, std::string("pre_compute_rope_table_node")};
            rope_table_node->set_node_meta(meta_rope_node);
            auto max_seq_len = this->_graph_ctx._max_seq_len;
            auto embedding_dim = this->_graph_ctx._n_embd_head;
            auto rope_base = this->_graph_ctx._rope_freq_base;
            if (this->_graph_ctx._rope_table == nullptr) {
                this->_graph_ctx._rope_table = std::make_shared<tff::core::memory::Tensor>(
                    memory::DataType::TFF_DATA_TYPE_F32,
                    memory::MemoryType::TFF_MEM_TYPE_RESIDENT,
                    std::array<int64_t, MAX_TENSOR_DIM>{
                        embedding_dim, max_seq_len, 1, 1
                    });
                auto device = tff::factory::ModuleFactory::instance()->create_shared<
                    tff::core::device::DeviceBaseObject>(
                    DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CUDA));
                std::vector<int> device_ids;
                device->get_device_id(device_ids);
                this->_graph_ctx._rope_table->set_allocator(device->get_device_buffer_allocator(device_ids[0]));
            }
            const auto &builder =
                    std::dynamic_pointer_cast<kernel::PreRopeTableBuilder>(rope_table_node->builder());
            builder->freqs(rope_base)
                    .max_seq_len(max_seq_len)
                    .hidden_dim(embedding_dim)
                    .rope_table(this->_graph_ctx._rope_table);
            rope_table_node->shape_infer();
        }
        return rope_table_node;
    }


    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_gather_node(std::shared_ptr<GraphNode> &node) {
        if (node == nullptr) {
            tff::log::Logger::error("[QWen3Creator::build_gather_node] node is null");
            return nullptr;
        }
        auto gather_node = ADD_NODE(TFF_OP_GATHER);
        gather_node->set_node_meta(NodeMetadata{node->name() + "_gather_node"});

        const auto &builder =
                std::dynamic_pointer_cast<kernel::GatherOPBuilder>(gather_node->builder());
        std::vector<int> row_indexs;
        row_indexs.resize(this->_graph_ctx._n_output);
        std::iota(row_indexs.begin(), row_indexs.end(),
                  this->_graph_ctx._n_tokens - this->_graph_ctx._n_output);
        builder->row_index(row_indexs).in(node->get_tensor());
        gather_node->shape_infer();

        gather_node->add_input_node(node);
        return gather_node;
    }

    std::shared_ptr<GraphNode> QWen3Creator::build_offload_node(std::shared_ptr<GraphNode> &node) {
        if (node == nullptr) {
            tff::log::Logger::error("[QWen3Creator::build_offload_node] node is null");
            return nullptr;
        }
        auto offload_node = ADD_NODE(TFF_OP_MEM_REF);
        offload_node->set_node_meta(NodeMetadata{false, true, node->name() + "_offload_node"});


        this->_graph_ctx._logits = std::make_shared<memory::Tensor>(node->get_tensor()->get_data_type(),
                                                                    memory::MemoryType::TFF_MEM_TYPE_RESIDENT,
                                                                    node->get_tensor()->get_shape());
        auto device_manager = std::dynamic_pointer_cast<device::DeviceManager>(
            tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                DEVICE_MANAGER_FLAG,
                tff::factory::ModuleKeyType(DEVICE_MANAGER_FLAG)));
        this->_graph_ctx._logits->set_allocator(device_manager->get_device(-1)
            ->get_device_buffer_allocator(-1));


        const auto &builder =
                std::dynamic_pointer_cast<kernel::MemRefBuilder>(offload_node->builder());
        builder->in(node->get_tensor()).out(this->_graph_ctx._logits);
        offload_node->shape_infer();
        offload_node->add_input_node(node);
        return offload_node;
    }

    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_rope_node(
        const int layer_id,
        const std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        if (input_node == nullptr) {
            tff::log::Logger::error("build_rope_node input node is null");
            return nullptr;
        }
        auto rope_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_ROPE);
        graph::NodeMetadata meta_q_rope_node{input_node->name() + "_rope"};
        rope_node->set_node_meta(meta_q_rope_node);
        const auto &builder =
                std::dynamic_pointer_cast<kernel::RopeBuilder>(rope_node->builder());
        builder->rope_type(static_cast<const int>(this->_graph_ctx._rope_type))
                .in(input_node->get_tensor());

        const auto &pos = this->_graph_ctx._pos;
        builder->token_idx(static_cast<int32_t*>(&pos->at<int32_t>(0)));

        if (layer_id == 0) {
            auto rope_table_node = build_rope_table_node();
            rope_node->add_input_node(rope_table_node);
        }
        builder->rope_table(this->_graph_ctx._rope_table);
        rope_node->shape_infer();
        rope_node->add_input_node(input_node);


        return rope_node;
    }

    //
    std::shared_ptr<GraphNode> QWen3Creator::build_inputs(
        const std::unordered_map<memory::ModelTensorType, std::shared_ptr<
            layer::ModelLayerObject> > &layer_map) {
        auto weight_layer = layer_map.find(
            memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD)->second;
        if (weight_layer == nullptr) {
            return nullptr;
        }

        if (layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN) == layer_map.end()) {
            tff::log::Logger::error("current batch has no valid input token layer");
            return nullptr;
        }
        auto input_token_layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN)->second;
        this->_graph_ctx._n_tokens = input_token_layer->_tensor->get_shape()[0];
        auto input_pos_layer = layer_map.find(memory::ModelTensorType::LLM_TENSOR_TOKEN_POS)->second;
        this->_graph_ctx._pos = input_pos_layer->_tensor;

        auto embedding_node = ADD_NODE(tff::core::graph::TffOpType::TFF_OP_EMBEDDING);
        const NodeMetadata meta_tokenize_node{true, false, "embedding"};
        embedding_node->set_node_meta(meta_tokenize_node);

        const auto &builder =
                std::dynamic_pointer_cast<tff::kernel::EmbeddingBuilder>(embedding_node->builder());
        builder->input_token(input_token_layer->_tensor).weight(weight_layer->_tensor);
        embedding_node->shape_infer();


        auto input_token_node = ADD_NODE(graph::TffOpType::TFF_OP_MEM_REF);
        input_token_node->set_node_meta(NodeMetadata{"input_token_node"});
        const auto &input_builder = std::dynamic_pointer_cast<kernel::MemRefBuilder>(input_token_node->builder());
        input_builder->in(input_token_layer->_tensor).out(input_token_layer->_tensor);
        input_token_node->shape_infer();

        embedding_node->add_input_node(
            this->_weight_node_map[LLM_TENSOR_LAYER_INPUT][0][memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD]);
        embedding_node->add_input_node(input_token_node);
        return embedding_node;
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_output(
        tff::core::memory::ModelTensorType tensor_type,
        std::shared_ptr<tff::core::graph::GraphNode> &input_node) {
        return build_mul_mat_node(
            this->_weight_node_map[LLM_TENSOR_LAYER_OUTPUT][0][tensor_type], input_node);
    }

    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_qkv_node(
        std::shared_ptr<tff::core::graph::GraphNode> &weight_node,
        std::shared_ptr<tff::core::graph::GraphNode> &x_node) {
        return build_mul_mat_node(weight_node,
                                  x_node);
    }


    //
    std::shared_ptr<tff::core::graph::GraphNode> QWen3Creator::build_ffn_inp(
        std::shared_ptr<tff::core::graph::GraphNode> &input_node,
        std::shared_ptr<tff::core::graph::GraphNode> &current_node) {
        if (input_node == nullptr || current_node == nullptr) {
            tff::log::Logger::error("build_ffn_inp input node is null");
            return nullptr;
        }
        auto ffn_inp = build_add_node(current_node, input_node);
        ffn_inp->set_node_meta(NodeMetadata{current_node->name() + "_add_inp"});
        return ffn_inp;
    }
}
