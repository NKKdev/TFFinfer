//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_TFFOPNODE_H
#define TFFINFER_TFFOPNODE_H
#include "graph/GraphNode.h"
#include "ModuleFactory.h"

#include "FunctionFactory.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/Buffer.h"

namespace tff::core::graph::op {
    class MatMulNode final : public tff::core::graph::GraphNode {
    public:
        explicit MatMulNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MUL_MAT);
        }

        ~MatMulNode() = default;

    public:
        template<typename T>
        auto get_xgemm_callback() {
            return tff::factory::FunctionFactory::instance()->get_callback(
                CREATE_LAYER_FLAG, tff::kernel::XGemm<T>::get_opthis->_node_metadata._name());
        }

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_MUL_MAT) {
                tff::log::Logger::error("MatMulNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            std::shared_ptr<core::memory::Tensor> _weight;
            std::shared_ptr<core::memory::Tensor> x;
            for (auto &input : this->input_nodes()) {
                if (input->op_type() == TFF_OP_MEM_REF) {
                    _weight = input->get_tensor();
                }else {
                    x = input->get_tensor();
                }
            }
            auto params = this->get_params();
            params->set_param(_weight);
            params->set_param(x);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };
    //
    class Q8MatMulNode final : public tff::core::graph::GraphNode {
    public:
        explicit Q8MatMulNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_QUANTIZE_Q8_MATMUL);
        }

        ~Q8MatMulNode() override = default;

    public:
        template<typename T>
        auto get_xgemm_callback() {
            return tff::factory::FunctionFactory::instance()->get_callback(
                CREATE_LAYER_FLAG, tff::kernel::XGemm<T>::get_opthis->_node_metadata._name());
        }

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_QUANTIZE_Q8_MATMUL) {
                tff::log::Logger::error("MatMulNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            std::shared_ptr<core::memory::Tensor> _weight;
            std::shared_ptr<core::memory::Tensor> x;
            for (auto &input : this->input_nodes()) {
                if (input->op_type() == TFF_OP_MEM_REF) {
                    _weight = input->get_tensor();
                }else {
                    x = input->get_tensor();
                }
            }
            auto params = this->get_params();
            params->set_param(_weight);
            params->set_param(x);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };
    //
    class UploadBuffer final : public tff::core::graph::GraphNode {
    public:
        explicit UploadBuffer(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_GET_ROWS);
        }

        ~UploadBuffer() override = default;

    public:
        template<typename T>
        auto get_read_buffer_callback() {
            return tff::factory::FunctionFactory::instance()->get_callback(
                CREATE_LAYER_FLAG, tff::kernel::XGemm<T>::get_opthis->_node_metadata._name());
        }

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("UploadBuffer param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_GET_ROWS) {
                tff::log::Logger::error("UploadBuffer op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class AddNode final : public tff::core::graph::GraphNode {
    public:
        explicit AddNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_ADD);
        };

        ~AddNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_ADD) {
                tff::log::Logger::error("AddNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class RMSNormNode final : public tff::core::graph::GraphNode {
    public:
        explicit RMSNormNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_RMS_NORM);
        }

        ~RMSNormNode() override = default;
    public:
        std::shared_ptr<core::memory::Tensor> _weight;// 可选;
        std::shared_ptr<core::memory::Tensor> _bias;// 可选;
        std::shared_ptr<core::memory::Tensor> _x;
    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_RMS_NORM) {
                tff::log::Logger::error("RMSNormNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->input_nodes().size() != 2) {
                tff::log::Logger::error("node (%s) params is invalid!!");
                return nullptr;
            }
            std::shared_ptr<core::memory::Tensor> weight;
            std::shared_ptr<core::memory::Tensor> x;
            for (auto &input : this->input_nodes()) {
                if (input->op_type() == TFF_OP_MEM_REF) {
                    weight = input->get_tensor();
                }else {
                    x = input->get_tensor();
                }
            }
            auto params = this->get_params();
            params->set_param(weight);
            params->set_param(x);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class NoneNode final : public tff::core::graph::GraphNode {
    public:
        explicit NoneNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_NONE);
        }

        ~NoneNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("NoneNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_NONE) {
                tff::log::Logger::error("NoneNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class DupNode final : public tff::core::graph::GraphNode {
    public:
        explicit DupNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_DUP);
        }


        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("DupNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_DUP) {
                tff::log::Logger::error("DupNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class SqrNode final : public tff::core::graph::GraphNode {
    public:
        explicit SqrNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_SQR);
        }

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SqrNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SQR) {
                tff::log::Logger::error("SqrNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class SqrtNode final : public tff::core::graph::GraphNode {
    public:
        explicit SqrtNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_SQRT);
        }

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SqrtNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SQRT) {
                tff::log::Logger::error("SqrtNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class SubNode final : public tff::core::graph::GraphNode {
    public:
        explicit SubNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_SUB); }

        ~SubNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SubNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SUB) {
                tff::log::Logger::error("SubNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class MulNode final : public tff::core::graph::GraphNode {
    public:
        explicit MulNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MUL); }

        ~MulNode() override = default;

        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_MUL) {
                tff::log::Logger::error("MulNode op type(expect TFF_OP_MUL) is wrong!!");
                return nullptr;
            }

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class DivNode final : public tff::core::graph::GraphNode {
    public:
        explicit DivNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_DIV); }

        ~DivNode() override = default;

        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("DivNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_DIV) {
                tff::log::Logger::error("DivNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class ReshapeNode final : public tff::core::graph::GraphNode {
    public:
        explicit ReshapeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_RESHAPE); }

        ~ReshapeNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_RESHAPE) {
                tff::log::Logger::error("ReshapeNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            for (auto &input : this->input_nodes()) {
                this->add_inputs(input->get_tensor());
            }
            auto params_ptr = this->get_params();
            params_ptr->set_param(this->_inputs);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class TransposeNode final : public tff::core::graph::GraphNode {
    public:
        explicit TransposeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_TRANSPOSE); }

        ~TransposeNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("TransposeNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_TRANSPOSE) {
                tff::log::Logger::error("TransposeNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class SoftmaxNode final : public tff::core::graph::GraphNode {
    public:
        explicit SoftmaxNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_SOFT_MAX); }

        ~SoftmaxNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SoftmaxNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SOFT_MAX) {
                tff::log::Logger::error("SoftmaxNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class RopeNode final : public tff::core::graph::GraphNode {
    public:
        explicit RopeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_ROPE); }

        ~RopeNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_ROPE) {
                tff::log::Logger::error("RopeNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class MapCPUBufferNode final : public tff::core::graph::GraphNode {
    public:
        explicit MapCPUBufferNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MAP2CPU); }

        ~MapCPUBufferNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            // if (this->_params_ptr->get_param_count() < 4) {
            //     tff::log::Logger::error("MapCPUBufferNode param count is %d(expect 1)",
            //         this->_params_ptr->get_param_count());
            //     return nullptr;
            // }
            for (auto &input : this->input_nodes()) {
                this->add_inputs(input->get_tensor());
            }
            auto params_ptr = this->get_params();
            params_ptr->set_param(this->_inputs);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class MemCpyNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemCpyNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MEM_CPY); }

        ~MemCpyNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_MEM_CPY) {
                tff::log::Logger::error("MemCpyNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            if (this->_params_ptr->get_param_count() < 2) {
                tff::log::Logger::error("MemCpyNode param count is less than 2");
                return nullptr;
            }
            for (auto &input : this->input_nodes()) {
                this->add_inputs(input->get_tensor());
            }
            auto params_ptr = this->get_params();
            params_ptr->set_param(this->_inputs);

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class EmbeddingNode final : public tff::core::graph::GraphNode {
    public:
        explicit EmbeddingNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_EMBEDDING); }

        ~EmbeddingNode() override = default;
    public:

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_EMBEDDING) {
                tff::log::Logger::error("EmbeddingNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            if (this->_input_nodes.size() != 2) {
                tff::log::Logger::error("error: EmbeddingNode input_node size(expect 2)");
                return nullptr;
            }
            std::shared_ptr<core::memory::Tensor> _embedding_weight;
            std::shared_ptr<core::memory::Tensor> _input_token;
            for (auto &input : this->_input_nodes) {
                if (input->get_tensor()->get_tensor_type() == tff::core::memory::ModelTensorType::LLM_TENSOR_TOKEN_EMBD) {
                    _embedding_weight = input->get_tensor();
                }
                if (input->get_tensor()->get_tensor_type() == tff::core::memory::ModelTensorType::LLM_TENSOR_INPUT_TOKEN) {
                    _input_token = input->get_tensor();
                }
            }
            auto params_ptr = this->get_params();
            params_ptr->set_param(_embedding_weight);
            params_ptr->set_param(_input_token);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class MemRefNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemRefNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MEM_REF);
        }

        ~MemRefNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_MEM_REF) {
                tff::log::Logger::error("MemRefNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

    //
    class FlashAttnNode final : public tff::core::graph::GraphNode {
    public:
        explicit FlashAttnNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_FLASH_ATTN_EXT); }

        ~FlashAttnNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_FLASH_ATTN_EXT) {
                tff::log::Logger::error("FlashAttnNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            std::shared_ptr<core::memory::Tensor> q;
            std::shared_ptr<core::memory::Tensor> k;
            std::shared_ptr<core::memory::Tensor> v;
            std::shared_ptr<core::memory::Tensor> rope_table;
            std::shared_ptr<core::memory::Tensor> mask;
            for (auto &input : this->_input_nodes) {
                auto tensor_type = input->get_tensor()->get_tensor_type();
                if (tensor_type == tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q) {
                    q = input->get_tensor();
                }else if (tensor_type == tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
                    k = input->get_tensor();
                }else if (tensor_type == tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                    v = input->get_tensor();
                }
                if (input->op_type() == TFF_OP_PRE_ROPE_TABLE) {
                    rope_table = input->get_tensor();
                }
                if (input->op_type() == TFF_OP_ATTN_MASK) {
                    mask = input->get_tensor();
                }
            }
            //
            // if (rope_table->get_buffer() == nullptr) {
            //     rope_table->set_buffer_data(this->_mem_manager_ptr->get_ptr_by_offset(this->_devices.begin()->first,
            //         this->_tensor->get_external_memory_index(), type),
            //         this->_tensor->get_bytes());
            //     rope_table->set_allocator(this->device().begin()->second->get_device_buffer_allocator(this->device().begin()->first));
            // }
            // if (mask->get_buffer() == nullptr) {
            //     mask->set_buffer_data(this->_mem_manager_ptr->get_ptr_by_offset(this->_devices.begin()->first,
            //         this->_tensor->get_external_memory_index(), type),
            //         this->_tensor->get_bytes());
            //     mask->set_allocator(this->device().begin()->second->get_device_buffer_allocator(this->device().begin()->first));
            // }

            auto params = this->get_params();
            params->set_param(q);
            params->set_param(k);
            params->set_param(v);
            params->set_param(rope_table);
            params->set_param(mask);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };
        //
    class PagedFlashAttnNode final : public tff::core::graph::GraphNode {
    public:
        explicit PagedFlashAttnNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_FLASH_ATTN_PAGED);
        }

        ~PagedFlashAttnNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_FLASH_ATTN_PAGED) {
                tff::log::Logger::error("PagedFlashAttnNode op type(expect TFF_OP_FLASH_ATTN_PAGED) is wrong!!");
                return nullptr;
            }
            std::shared_ptr<core::memory::Tensor> q;
            std::shared_ptr<core::memory::Tensor> k;
            std::shared_ptr<core::memory::Tensor> v;
            std::shared_ptr<core::memory::Tensor> rope_table;
            std::shared_ptr<core::memory::Tensor> mask;
            for (auto &input : this->_input_nodes) {
                auto tensor_type = input->get_tensor()->get_tensor_type();
                if (tensor_type == tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q) {
                    q = input->get_tensor();
                }else if (tensor_type == tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
                    k = input->get_tensor();
                }else if (tensor_type == tff::core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                    v = input->get_tensor();
                }
                if (input->op_type() == TFF_OP_PRE_ROPE_TABLE) {
                    rope_table = input->get_tensor();
                }
                if (input->op_type() == TFF_OP_ATTN_MASK) {
                    mask = input->get_tensor();
                }
            }

            auto params = this->get_params();
            params->set_param(q);
            params->set_param(k);
            params->set_param(v);
            params->set_param(rope_table);
            params->set_param(mask);
            auto callback = GraphNode::forward(type);
            return callback;
        }
    };
    //
    class PreRopeTableNode final : public tff::core::graph::GraphNode {
    public:
        explicit PreRopeTableNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_PRE_ROPE_TABLE); }

        ~PreRopeTableNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_PRE_ROPE_TABLE) {
                tff::log::Logger::error("PreRopeTableNode op type(expect TFF_OP_PRE_ROPE_TABLE) is wrong!!");
                return nullptr;
            }

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };
    //
    class UnaryOPNode final : public tff::core::graph::GraphNode {
        public:
        explicit UnaryOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_UNARY);
        }
        ~UnaryOPNode() override = default;
    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_UNARY) {
                tff::log::Logger::error("UnaryOPNode op type(expect TFF_OP_UNARY) is wrong!!");
                return nullptr;
            }
            std::shared_ptr<core::memory::Tensor> ffn_up_x;
            std::shared_ptr<core::memory::Tensor> ffn_gate_x;
            for (auto &input : this->_input_nodes) {
                if (input->get_tensor()->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_FFN_UP) {
                    ffn_up_x = input->get_tensor();
                }
                if (input->get_tensor()->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_FFN_GATE) {
                    ffn_gate_x = input->get_tensor();
                }
            }
            auto params = this->get_params();
            params->set_param(ffn_gate_x);
            params->set_param(ffn_up_x);
            auto callback = GraphNode::forward(type);
            return callback;
        }

    };
    //
    class MaskOPNode final : public tff::core::graph::GraphNode {
    public:
        explicit MaskOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_ATTN_MASK);
        }
        ~MaskOPNode() override = default;
    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_ATTN_MASK) {
                tff::log::Logger::error("MaskOPNode op type(expect TFF_OP_ATTN_MASK) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward(type);
            return callback;
        }

    };
    //
    class GetOfRowsOPNode final : public tff::core::graph::GraphNode {
        public:
        explicit GetOfRowsOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_GET_ROWS);
        }
        ~GetOfRowsOPNode() override = default;
        public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_GET_ROWS) {
                tff::log::Logger::error("GetOfRowsOPNode op type (expect TFF_OP_GET_ROWS) is wrong");
                return nullptr;
            }

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };
    //
    class SetOfRowsOPNode final : public tff::core::graph::GraphNode {
        public:
        explicit SetOfRowsOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_SET_ROWS);
        }
        ~SetOfRowsOPNode() override = default;
        public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward(runtime::MemoryType &type) override {
            if (this->_op_type != TFF_OP_SET_ROWS) {
                tff::log::Logger::error("SetOfRowsOPNode op type (expect TFF_OP_SET_ROWS) is wrong");
                return nullptr;
            }
            std::shared_ptr<core::memory::Tensor> k_tensor;
            std::shared_ptr<core::memory::Tensor> v_tensor;
            for (auto &input : this->_input_nodes) {
                if (input->get_tensor()->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
                    k_tensor = input->get_tensor();
                }
                if (input->get_tensor()->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                    v_tensor = input->get_tensor();
                }
            }
            auto params = this->get_params();
            params->set_param(k_tensor);
            params->set_param(v_tensor);

            auto callback = GraphNode::forward(type);
            return callback;
        }
    };

}


#endif //TFFINFER_TFFOPNODE_H
