//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_TFFOPNODE_H
#define TFFINFER_TFFOPNODE_H
#include "graph/GraphNode.h"
#include "ModuleFactory.h"
#include "global/GlobalDefine.h"
#include "FunctionFactory.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/Buffer.h"

namespace tff::core::graph::op {
    class MatMulNode final : public tff::core::graph::GraphNode {
    public:
        explicit MatMulNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MUL_MAT);
        }

        ~MatMulNode() override = default;

    public:
        template<typename T>
        auto get_xgemm_callback() {
            return tff::factory::FunctionFactory::instance()->get_callback(
                CREATE_LAYER_FLAG, tff::kernel::XGemm<T>::get_opthis->_node_metadata._name());
        }

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_MUL_MAT) {
                tff::log::Logger::error("MatMulNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_prev_nodes.size() != 2) {
                tff::log::Logger::error("MatMulNode previous node count is wrong!!");
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }

            auto input_tensor_0 = *this->_src_tensors_ptr.begin();
            auto input_tensor_1 = *this->_src_tensors_ptr.rbegin();
            std::array<int64_t, MAX_TENSOR_DIM> output_shape{input_tensor_1->get_shape()[0], input_tensor_0->get_shape()[1], 1, 1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(output_shape.size(), input_tensor_0->get_data_type(),
                                                                             output_shape, true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});

            auto callback = GraphNode::forward();
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
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("UploadBuffer param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_GET_ROWS) {
                tff::log::Logger::error("UploadBuffer op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
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
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_ADD) {
                tff::log::Logger::error("AddNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_prev_nodes.size() != 2) {
                tff::log::Logger::error("MatMulNode previous node count is wrong!!");
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }

            auto input_tensor_0 = *this->_src_tensors_ptr.begin();
            auto input_tensor_1 = *this->_src_tensors_ptr.rbegin();
            std::array<int64_t, MAX_TENSOR_DIM> output_shape{input_tensor_0->get_shape()[0], input_tensor_0->get_shape()[1], 1, 1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(output_shape.size(), input_tensor_0->get_data_type(),
                                                                             output_shape, true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
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
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_RMS_NORM) {
                tff::log::Logger::error("RMSNormNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }
            if (this->_src_tensors_ptr.empty()) {
                tff::log::Logger::error("current node(%s) has invalid input", this->_node_metadata._name.c_str());
                return nullptr;
            }
            auto input_tensor_x = *this->_src_tensors_ptr.begin();
            auto input_tensor_weight = *this->_src_tensors_ptr.rbegin();
            if (input_tensor_x == nullptr) {
                tff::log::Logger::error("input tensor x is null");
                return nullptr;
            }
            if (input_tensor_weight == nullptr) {
                tff::log::Logger::error("input tensor weight is null");
                return nullptr;
            }
            if (input_tensor_x->get_shape().empty() || input_tensor_weight->get_shape().empty()) {
                tff::log::Logger::error("input tensor shape is empty");
                return nullptr;
            }
            if (input_tensor_x->get_shape().size() < 2) {
                tff::log::Logger::error("input tensor shape is not equal to 2");
                return nullptr;
            }
            std::array<int64_t, MAX_TENSOR_DIM> dst_shape{input_tensor_weight->get_shape()[0], input_tensor_x->get_shape()[1], 1, 1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(dst_shape.size(),input_tensor_x->get_data_type(),
                dst_shape,true, input_tensor_x->get_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
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
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("NoneNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_NONE) {
                tff::log::Logger::error("NoneNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class DupNode final : public tff::core::graph::GraphNode {
    public:
        explicit DupNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_DUP);
        }


        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("DupNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_DUP) {
                tff::log::Logger::error("DupNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
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
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SqrNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SQR) {
                tff::log::Logger::error("SqrNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
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
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SqrtNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SQRT) {
                tff::log::Logger::error("SqrtNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class SubNode final : public tff::core::graph::GraphNode {
    public:
        explicit SubNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_SUB); }

        ~SubNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SubNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SUB) {
                tff::log::Logger::error("SubNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class MulNode final : public tff::core::graph::GraphNode {
    public:
        explicit MulNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MUL); }

        ~MulNode() override = default;

        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_MUL) {
                tff::log::Logger::error("MulNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_prev_nodes.size() != 2) {
                tff::log::Logger::error("MatMulNode previous node count is wrong!!");
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }

            auto input_tensor_0 = *this->_src_tensors_ptr.begin();
            auto input_tensor_1 = *this->_src_tensors_ptr.rbegin();
            std::array<int64_t, MAX_TENSOR_DIM> output_shape{input_tensor_0->get_shape()[0], input_tensor_0->get_shape()[1], 1, 1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(output_shape.size(), input_tensor_0->get_data_type(),
                                                                             output_shape, true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class DivNode final : public tff::core::graph::GraphNode {
    public:
        explicit DivNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_DIV); }

        ~DivNode() override = default;

        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("DivNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_DIV) {
                tff::log::Logger::error("DivNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class ReshapeNode final : public tff::core::graph::GraphNode {
    public:
        explicit ReshapeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_RESHAPE); }

        ~ReshapeNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_RESHAPE) {
                tff::log::Logger::error("ReshapeNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_prev_nodes.size() != 1) {
                tff::log::Logger::error("MatMulNode previous node count is wrong!!");
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }

            auto input_tensor_0 = *this->_src_tensors_ptr.begin();
            std::array<int64_t, MAX_TENSOR_DIM> output_shape{input_tensor_0->get_shape()[0], input_tensor_0->get_shape()[1], 1, 1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(output_shape.size(), input_tensor_0->get_data_type(),
                                                                             output_shape, true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class TransposeNode final : public tff::core::graph::GraphNode {
    public:
        explicit TransposeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_TRANSPOSE); }

        ~TransposeNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("TransposeNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_TRANSPOSE) {
                tff::log::Logger::error("TransposeNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class SoftmaxNode final : public tff::core::graph::GraphNode {
    public:
        explicit SoftmaxNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_SOFT_MAX); }

        ~SoftmaxNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_params_ptr->get_param_count() < 4) {
                tff::log::Logger::error("SoftmaxNode param count is %d(expect 1)",
                                        this->_params_ptr->get_param_count());
                return nullptr;
            }
            if (this->_op_type != TFF_OP_SOFT_MAX) {
                tff::log::Logger::error("SoftmaxNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class RopeNode final : public tff::core::graph::GraphNode {
    public:
        explicit RopeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_ROPE); }

        ~RopeNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_ROPE) {
                tff::log::Logger::error("RopeNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_prev_nodes.size() != 1) {
                tff::log::Logger::error("MatMulNode previous node count is wrong!!");
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }

            auto input_tensor_0 = *this->_src_tensors_ptr.begin();
            std::array<int64_t, MAX_TENSOR_DIM> output_shape{input_tensor_0->get_shape()[0], input_tensor_0->get_shape()[1], 1, 1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(output_shape.size(), input_tensor_0->get_data_type(),
                                                                             output_shape, true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class MapCPUBufferNode final : public tff::core::graph::GraphNode {
    public:
        explicit MapCPUBufferNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MAP2CPU); }

        ~MapCPUBufferNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            // if (this->_params_ptr->get_param_count() < 4) {
            //     tff::log::Logger::error("MapCPUBufferNode param count is %d(expect 1)",
            //         this->_params_ptr->get_param_count());
            //     return nullptr;
            // }
            if (this->_op_type != TFF_OP_MAP2CPU) {
                tff::log::Logger::error("MapCPUBufferNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_src_tensors_ptr.size() != 1) {
                tff::log::Logger::error("MapCPUBufferNode src_tensors size(%d) is wrong!!",this->_src_tensors_ptr.size());
                return nullptr;
            }
            this->_dst_tensors_ptr.insert(this->_dst_tensors_ptr.end(),this->_src_tensors_ptr.begin(), this->_src_tensors_ptr.end());

            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class MemCpyNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemCpyNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MEM_CPY); }

        ~MemCpyNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_MEM_CPY) {
                tff::log::Logger::error("MemCpyNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }

            if (this->_params_ptr->get_param_count() < 2) {
                tff::log::Logger::error("MemCpyNode param count is less than 2");
                return nullptr;
            }
            for (auto &pre_node : this->_prev_nodes) {
                auto output_tensor = pre_node.lock()->outputs();
                this->set_inputs(output_tensor);
            }
            // auto memcpy_kind_opt = this->_params_ptr->get_param<int>(1);
            // if (!memcpy_kind_opt.has_value()) {
            //     tff::log::Logger::error("MemCpyNode memcpy_kind_opt has no value");
            //     return nullptr;
            // }
            if (this->_src_tensors_ptr.size() != 1) {
                tff::log::Logger::error("MemCpyNode src_tensors size(%d) is wrong!!",
                    this->_src_tensors_ptr.size());
                return nullptr;
            }
            auto &input_tensors = *this->_src_tensors_ptr.begin();
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(input_tensors->get_shape().size(), input_tensors->get_data_type(),
                                                                             input_tensors->get_shape(), true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class EmbeddingNode final : public tff::core::graph::GraphNode {
    public:
        explicit EmbeddingNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_EMBEDDING); }

        ~EmbeddingNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_EMBEDDING) {
                tff::log::Logger::error("EmbeddingNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_prev_nodes.size() != 2) {
                tff::log::Logger::error("EmbeddingNode prev_nodes size(%d) is wrong!!",this->_prev_nodes.size());
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }
            auto input_tensor_0 = *this->_src_tensors_ptr.begin();
            auto input_tensor_1 = *this->_src_tensors_ptr.rbegin();
            std::array<int64_t, MAX_TENSOR_DIM> output_shape{input_tensor_0->get_shape()[0], input_tensor_1->get_shape()[0],1,1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(output_shape.size(), memory::DataType::TFF_DATA_TYPE_F32,
                                                                             output_shape, true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
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
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_MEM_REF) {
                tff::log::Logger::error("MemRefNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_src_tensors_ptr.size() != 1) {
                tff::log::Logger::error("MemRefNode src_tensors size() is wrong!!");
                return nullptr;
            }
            this->set_outputs(this->_src_tensors_ptr);
            auto callback = GraphNode::forward();
            return callback;
        }
    };

    //
    class FlashAttnNode final : public tff::core::graph::GraphNode {
    public:
        explicit FlashAttnNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_FLASH_ATTN_EXT); }

        ~FlashAttnNode() override = default;

    public:
        std::function<tff::kernel::base::OP_CALLBACK_TYPE> forward() override {
            if (this->_op_type != TFF_OP_FLASH_ATTN_EXT) {
                tff::log::Logger::error("FlashAttnNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return nullptr;
            }
            if (this->_prev_nodes.size() != 3) {
                tff::log::Logger::error("FlashAttnNode prev_nodes size() is wrong!!");
                return nullptr;
            }
            for (auto &prev_node : this->_prev_nodes) {
                auto pre_output_tensor = prev_node.lock()->outputs();
                this->set_inputs(pre_output_tensor);
            }
            auto input_tensor_0 = *this->_src_tensors_ptr.begin();
            auto input_tensor_1 = *this->_src_tensors_ptr.rbegin();
            std::array<int64_t, MAX_TENSOR_DIM> output_shape{input_tensor_0->get_shape()[0], input_tensor_0->get_shape()[1],1,1};
            auto output_tensor = std::make_shared<tff::core::memory::Tensor>(output_shape.size(),input_tensor_0->get_data_type(),
                                                                             output_shape, true,
                                                                             this->device()->
                                                                             get_device_buffer_allocator());
            this->set_outputs(std::vector<std::shared_ptr<tff::core::memory::Tensor>>{output_tensor});
            auto callback = GraphNode::forward();
            return callback;
        }
    };
}


#endif //TFFINFER_TFFOPNODE_H
