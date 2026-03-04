//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_TFFOPNODE_H
#define TFFINFER_TFFOPNODE_H
#include "graph/GraphNode.h"
#include "ModuleFactory.h"

#include "FunctionFactory.h"
#include "device/DeviceManager.h"
#include "kernel/include/TFFOPCreator.h"
#include "runtime/KVCache.h"

namespace tff::core::graph::op {
    class MatMulNode final : public tff::core::graph::GraphNode {
    public:
        explicit MatMulNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MUL_MAT);
            this->_builder = std::make_shared<kernel::MatMulBuilder>();
        }

        ~MatMulNode() = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_MUL_MAT) {
                tff::log::Logger::error("MatMulNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::MatMulBuilder>(this->_builder);
            auto a = builder->a<std::shared_ptr<core::memory::Tensor> >();
            auto b = builder->b<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::MatMulBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MatMulBuilder) is wrong!!");
                return;
            }
            auto a = builder->a<std::shared_ptr<core::memory::Tensor> >();
            auto b = builder->b<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(a, kernel::MatMulBuilder::Params::A));
            this->_tensor_param_map.insert(std::make_pair(b, kernel::MatMulBuilder::Params::B));
            std::array<int64_t, MAX_TENSOR_DIM> shape = {
                a->get_shape()[1], b->get_shape()[1], 1, 1
            };
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<tff::core::memory::Tensor>(
                    memory::DataType::TFF_DATA_TYPE_F32,
                    memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                    shape);
                this->_tensor->set_tensor_type(a->get_tensor_type());
                this->_tensor->set_allocator(a->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class QuantMatMulNode final : public tff::core::graph::GraphNode {
    public:
        explicit QuantMatMulNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_QUANTIZE_MATMUL);
            this->_builder = std::make_shared<kernel::QuantMatMulBuilder>();
        }

        ~QuantMatMulNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_QUANTIZE_MATMUL) {
                tff::log::Logger::error("QuantMatMulNode op type(expect TFF_OP_QUANTIZE_MATMUL) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::QuantMatMulBuilder>(this->_builder);
            auto weight = builder->weight<std::shared_ptr<core::memory::Tensor> >();
            auto x = builder->x<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::QuantMatMulBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MatMulBuilder) is wrong!!");
                return;
            }
            auto weight = builder->weight<std::shared_ptr<core::memory::Tensor> >();
            auto x = builder->x<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(weight, kernel::QuantMatMulBuilder::Params::Weight));
            this->_tensor_param_map.insert(std::make_pair(x, kernel::QuantMatMulBuilder::Params::X));
            std::array<int64_t, MAX_TENSOR_DIM> shape = {
                weight->get_shape()[1], x->get_shape()[1], 1, 1
            };
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<tff::core::memory::Tensor>(
                    memory::DataType::TFF_DATA_TYPE_F32,
                    memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                    shape);
                this->_tensor->set_tensor_type(weight->get_tensor_type());
                this->_tensor->set_allocator(weight->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class AddNode final : public tff::core::graph::GraphNode {
    public:
        explicit AddNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_ADD);
            this->_builder = std::make_shared<kernel::AddBuilder>();
        };

        ~AddNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_ADD) {
                tff::log::Logger::error("AddNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }
            if (this->input_nodes().size() != 2) {
                tff::log::Logger::error("AddNode (%s) params is invalid!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::AddBuilder>(this->_builder);
            auto x1 = builder->x1<std::shared_ptr<core::memory::Tensor> >();
            auto x2 = builder->x2<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::AddBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MatMulBuilder) is wrong!!");
                return;
            }
            auto x1 = builder->x1<std::shared_ptr<core::memory::Tensor> >();
            auto x2 = builder->x2<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(x1, kernel::AddBuilder::Params::X1));
            this->_tensor_param_map.insert(std::make_pair(x2, kernel::AddBuilder::Params::X2));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(x1);
                this->_tensor->set_allocator(x1->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class RMSNormNode final : public tff::core::graph::GraphNode {
    public:
        explicit RMSNormNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_RMS_NORM);
            this->_builder = std::make_shared<kernel::NormBuilder>();
        }

        ~RMSNormNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_RMS_NORM) {
                tff::log::Logger::error("RMSNormNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::NormBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::NormBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MatMulBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::NormBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(
                    memory::DataType::TFF_DATA_TYPE_F32,
                    memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                    input_tensor->get_shape());
                this->_tensor->set_tensor_type(input_tensor->get_tensor_type());
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class RMSNormWNode final : public tff::core::graph::GraphNode {
    public:
        explicit RMSNormWNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_RMS_NORM_W);
            this->_builder = std::make_shared<kernel::NormBuilder>();
        }

        ~RMSNormWNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_RMS_NORM_W) {
                tff::log::Logger::error("RMSNormNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::NormBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::NormBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MatMulBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::NormBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = builder->out<std::shared_ptr<core::memory::Tensor> >();
                builder->out(this->_tensor);
            }
        }
    };

    //
    class MulNode final : public tff::core::graph::GraphNode {
    public:
        explicit MulNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MUL);
            this->_builder = std::make_shared<kernel::MulBuilder>();
        }

        ~MulNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_MUL) {
                tff::log::Logger::error("MulNode op type(expect TFF_OP_MUL) is wrong!!");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::MulBuilder>(this->_builder);
            auto weight = builder->weight<const std::shared_ptr<memory::Tensor>>();
            auto x = builder->x<const std::shared_ptr<memory::Tensor>>();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::MulBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MulBuilder) is wrong!!");
                return;
            }
            auto weight = builder->weight<const std::shared_ptr<memory::Tensor>>();
            auto x = builder->x<const std::shared_ptr<memory::Tensor>>();
            this->_tensor_param_map.insert(std::make_pair(weight, kernel::MulBuilder::Params::Weight));
            this->_tensor_param_map.insert(std::make_pair(x, kernel::MulBuilder::Params::X));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(
                    memory::DataType::TFF_DATA_TYPE_F32,
                    memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                    x->get_shape());
                this->_tensor->set_tensor_type(weight->get_tensor_type());
                this->_tensor->set_allocator(x->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class ReshapeNode final : public tff::core::graph::GraphNode {
    public:
        explicit ReshapeNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_RESHAPE);
            this->_builder = std::make_shared<kernel::ReshapeBuilder>();
        }

        ~ReshapeNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_RESHAPE) {
                tff::log::Logger::error("ReshapeNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }
            if (this->input_nodes().size() != 1) {
                tff::log::Logger::error("ReshapeNode op expect 1 input!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::ReshapeBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::ReshapeBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect ReshapeBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::ReshapeBuilder::Params::In));
            std::array<int64_t, MAX_TENSOR_DIM> shape = {
                builder->embd_head_num<const int>(), builder->head_num<const int>(), builder->token_num<const int>(), 1
            };
            if (this->_tensor == nullptr) {
                this->_tensor = input_tensor;
                this->_tensor->set_shape(shape);
                builder->out(this->_tensor);
            }
        }
    };


    //`
    class RopeNode final : public tff::core::graph::GraphNode {
    public:
        explicit RopeNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_ROPE);
            this->_builder = std::make_shared<kernel::RopeBuilder>();
        }

        ~RopeNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_ROPE) {
                tff::log::Logger::error("RopeNode op type(expect TFF_OP_ROPE) is wrong!!");
                return;
            }
            if (this->input_nodes().size() != 1) {
                tff::log::Logger::error("RopeNode input_nodes size (expect 1)");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::RopeBuilder>(this->_builder);
            auto rope_table = builder->rope_table<std::shared_ptr<core::memory::Tensor> >();
            auto rope_type = builder->rope_type<const int>();
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::RopeBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect RopeBuilder) is wrong!!");
                return;
            }
            auto rope_table = builder->rope_table<std::shared_ptr<core::memory::Tensor> >();
            auto rope_type = builder->rope_type<const int>();
            auto token_idx = builder->token_idx<int32_t*>();
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::RopeBuilder::Params::In));
            this->_tensor_param_map.insert(std::make_pair(rope_table, kernel::RopeBuilder::Params::RopeTable));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<core::memory::Tensor>(memory::DataType::TFF_DATA_TYPE_F32,
                                                                       memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                       input_tensor->get_shape());
                this->_tensor->set_tensor_type(input_tensor->get_tensor_type());
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class MapCPUBufferNode final : public tff::core::graph::GraphNode {
    public:
        explicit MapCPUBufferNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MAP2CPU);
            this->_builder = std::make_shared<kernel::Map2CpuBuilder>();
        }

        ~MapCPUBufferNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_MAP2CPU) {
                tff::log::Logger::error("MapCPUBufferNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::Map2CpuBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::Map2CpuBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect Map2CpuBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::Map2CpuBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<core::memory::Tensor>(input_tensor);

                auto device = tff::factory::ModuleFactory::instance()->create_shared<
                    tff::core::device::DeviceBaseObject>(
                    DEVICE_BACKEND_FLAG, tff::factory::ModuleKeyType(DEVICE_BACKEND_TYPE_CPU));
                std::vector<int> device_ids;
                device->get_device_id(device_ids);
                this->_tensor->set_allocator(device->get_device_buffer_allocator(device_ids[0]));
                builder->out(this->_tensor);
            }
        }
    };

    //
    class MemCpyNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemCpyNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MEM_CPY);
            this->_builder = std::make_shared<kernel::MemCpyBuilder>();
        }

        ~MemCpyNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_MEM_CPY) {
                tff::log::Logger::error("MemCpyNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }
            std::vector<std::shared_ptr<core::memory::Tensor> > inputs;
            for (auto &input: this->_input_nodes) {
                inputs.push_back(input->get_tensor());
            }

            auto builder = std::dynamic_pointer_cast<kernel::MemCpyBuilder>(this->_builder);
            auto mem_cpy_kind = static_cast<memory::MemCpyKind>(builder->memcpy_kind<core::memory::MemCpyKind>());
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::MemCpyBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MemCpyBuilder) is wrong!!");
                return;
            }
            auto mem_cpy_kind = static_cast<memory::MemCpyKind>(builder->memcpy_kind<core::memory::MemCpyKind>());
            auto source_device_id = builder->source_id<int>();
            auto dest_device_id = builder->dest_id<int>();
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::MemCpyBuilder::Params::In));
            auto out_tensor = builder->out<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::MemCpyBuilder::Params::In));
            if (this->_tensor == nullptr) {
                if (out_tensor != nullptr) {
                    this->_tensor = out_tensor;
                } else {
                    this->_tensor = std::make_shared<memory::Tensor>(input_tensor);
                    builder->out(this->_tensor);
                }
            }
            auto device_manager = std::dynamic_pointer_cast<device::DeviceManager>(
                tff::factory::ModuleFactory::instance()->create_shared<tff::module::ModuleObject>(
                    DEVICE_MANAGER_FLAG,
                    tff::factory::ModuleKeyType(DEVICE_MANAGER_FLAG)));
            this->_tensor->set_allocator(device_manager->get_device(dest_device_id)
                ->get_device_buffer_allocator(dest_device_id));
        }
    };

    //
    class EmbeddingNode final : public tff::core::graph::GraphNode {
    public:
        explicit EmbeddingNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_EMBEDDING);
            this->_builder = std::make_shared<kernel::EmbeddingBuilder>();
        }

        ~EmbeddingNode() override = default;

    public:

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_EMBEDDING) {
                tff::log::Logger::error("EmbeddingNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::EmbeddingBuilder>(this->_builder);
            auto input_token = builder->input_token<const std::shared_ptr<memory::Tensor>>();
            auto weight = builder->weight<const std::shared_ptr<memory::Tensor>>();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::EmbeddingBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect EmbeddingBuilder) is wrong!!");
                return;
            }
            auto input_token = builder->input_token<const std::shared_ptr<memory::Tensor>>();
            auto weight = builder->weight<const std::shared_ptr<memory::Tensor>>();
            this->_tensor_param_map.insert(std::make_pair(input_token, kernel::EmbeddingBuilder::Params::InputToken));
            this->_tensor_param_map.insert(std::make_pair(weight, kernel::EmbeddingBuilder::Params::Weight));
            std::array<int64_t, MAX_TENSOR_DIM> shape = {
                weight->get_shape()[0], input_token->get_shape()[0],
                input_token->get_shape()[1], 1
            };
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<tff::core::memory::Tensor>(
                    memory::DataType::TFF_DATA_TYPE_F32,
                    memory::MemoryType::TFF_MEM_TYPE_WORKSPACE, shape);
                this->_tensor->set_tensor_type(weight->get_tensor_type());
                this->_tensor->set_allocator(weight->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class MemRefNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemRefNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MEM_REF);
            this->_builder = std::make_shared<kernel::MemRefBuilder>();
        }

        ~MemRefNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_MEM_REF) {
                tff::log::Logger::error("MemRefNode op type(expect TFF_OP_MEM_REF) is wrong!!");
                return;
            }


            auto builder = std::dynamic_pointer_cast<kernel::MemRefBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::MemRefBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MemRefBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            auto output_tensor = builder->out<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::MemRefBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = output_tensor;
                builder->out(this->_tensor);
            }
        }
    };

    //
    class FlashAttnNode final : public tff::core::graph::GraphNode {
    public:
        explicit FlashAttnNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_FLASH_ATTN_EXT);
            this->_builder = std::make_shared<kernel::FlashAttnBuilder>();
        }

        ~FlashAttnNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_FLASH_ATTN_EXT) {
                tff::log::Logger::error("FlashAttnNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::FlashAttnBuilder>(this->_builder);
            auto q = builder->q<std::shared_ptr<core::memory::Tensor> >();
            auto k = builder->k<std::shared_ptr<core::memory::Tensor> >();
            auto v = builder->v<std::shared_ptr<core::memory::Tensor> >();
            auto mask = builder->mask<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::FlashAttnBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect FlashAttnBuilder) is wrong!!");
                return;
            }
            auto q = builder->q<std::shared_ptr<core::memory::Tensor> >();
            auto k = builder->k<std::shared_ptr<core::memory::Tensor> >();
            auto v = builder->v<std::shared_ptr<core::memory::Tensor> >();
            auto mask = builder->mask<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(q, kernel::FlashAttnBuilder::Params::Q));
            this->_tensor_param_map.insert(std::make_pair(k, kernel::FlashAttnBuilder::Params::K));
            this->_tensor_param_map.insert(std::make_pair(v, kernel::FlashAttnBuilder::Params::V));
            this->_tensor_param_map.insert(std::make_pair(mask, kernel::FlashAttnBuilder::Params::Mask));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(memory::DataType::TFF_DATA_TYPE_F32,
                                                                 memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                 q->get_shape());
                this->_tensor->set_allocator(q->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class PagedFlashAttnNode final : public tff::core::graph::GraphNode {
    public:
        explicit PagedFlashAttnNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_FLASH_ATTN_PAGED);
            this->_builder = std::make_shared<kernel::PagedFlashAttnBuilder>();
        }

        ~PagedFlashAttnNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_FLASH_ATTN_PAGED) {
                tff::log::Logger::error("PagedFlashAttnNode op type(expect TFF_OP_FLASH_ATTN_PAGED) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::PagedFlashAttnBuilder>(this->_builder);
            auto q = builder->q<std::shared_ptr<core::memory::Tensor> >();
            auto k = builder->k<std::shared_ptr<core::memory::Tensor> >();
            auto v = builder->v<std::shared_ptr<core::memory::Tensor> >();
            auto mask = builder->mask<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::PagedFlashAttnBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect PagedFlashAttnBuilder) is wrong!!");
                return;
            }
            auto q = builder->q<std::shared_ptr<core::memory::Tensor> >();
            auto k = builder->k<std::shared_ptr<core::memory::Tensor> >();
            auto v = builder->v<std::shared_ptr<core::memory::Tensor> >();
            auto mask = builder->mask<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(q, kernel::PagedFlashAttnBuilder::Params::Q));
            this->_tensor_param_map.insert(std::make_pair(k, kernel::PagedFlashAttnBuilder::Params::K));
            this->_tensor_param_map.insert(std::make_pair(v, kernel::PagedFlashAttnBuilder::Params::V));
            this->_tensor_param_map.insert(std::make_pair(mask, kernel::PagedFlashAttnBuilder::Params::Mask));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(memory::DataType::TFF_DATA_TYPE_F32,
                                                                 memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                 q->get_shape());
                this->_tensor->set_allocator(q->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class PagedFlashAttnRopeNode final : public tff::core::graph::GraphNode {
    public:
        explicit PagedFlashAttnRopeNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_FLASH_ATTN_PAGED_ROPE);
            this->_builder = std::make_shared<kernel::PagedFlashAttnBuilder>();
        }

        ~PagedFlashAttnRopeNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_FLASH_ATTN_PAGED_ROPE) {
                tff::log::Logger::error("PagedFlashAttnNode op type(expect TFF_OP_FLASH_ATTN_PAGED) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::PagedFlashAttnBuilder>(this->_builder);
            auto q = builder->q<std::shared_ptr<core::memory::Tensor> >();
            auto k = builder->k<std::shared_ptr<core::memory::Tensor> >();
            auto v = builder->v<std::shared_ptr<core::memory::Tensor> >();
            auto mask = builder->mask<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::PagedFlashAttnBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect PagedFlashAttnBuilder) is wrong!!");
                return;
            }
            auto q = builder->q<std::shared_ptr<core::memory::Tensor> >();
            auto k = builder->k<std::shared_ptr<core::memory::Tensor> >();
            auto v = builder->v<std::shared_ptr<core::memory::Tensor> >();
            auto mask = builder->mask<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(q, kernel::PagedFlashAttnBuilder::Params::Q));
            this->_tensor_param_map.insert(std::make_pair(k, kernel::PagedFlashAttnBuilder::Params::K));
            this->_tensor_param_map.insert(std::make_pair(v, kernel::PagedFlashAttnBuilder::Params::V));
            this->_tensor_param_map.insert(std::make_pair(mask, kernel::PagedFlashAttnBuilder::Params::Mask));
            if (this->_tensor == nullptr) {
                this->_tensor = builder->out<std::shared_ptr<memory::Tensor> >();
                builder->out(this->_tensor);
            }
        }
    };

    //
    class PreRopeTableNode final : public tff::core::graph::GraphNode {
    public:
        explicit PreRopeTableNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_PRE_ROPE_TABLE);
            this->_builder = std::make_shared<kernel::PreRopeTableBuilder>();
        }

        ~PreRopeTableNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_PRE_ROPE_TABLE) {
                tff::log::Logger::error("PreRopeTableNode op type(expect TFF_OP_PRE_ROPE_TABLE) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::PreRopeTableBuilder>(this->_builder);
            auto table = builder->rope_table<std::shared_ptr<core::memory::Tensor> >();
            auto max_seq_len = builder->max_seq_len<int>();
            auto embedding_dim = builder->hidden_dim<int>();
            auto rope_base = builder->freqs<float>();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::PreRopeTableBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect PreRopeTableBuilder) is wrong!!");
                return;
            }
            auto table = builder->rope_table<std::shared_ptr<core::memory::Tensor> >();
            auto max_seq_len = builder->max_seq_len<int>();
            auto embedding_dim = builder->hidden_dim<int>();
            auto rope_base = builder->freqs<float>();
            this->_tensor_param_map.insert(std::make_pair(table, kernel::PreRopeTableBuilder::Params::RopeTable));
            if (this->_tensor == nullptr) {
                this->_tensor = table;
                builder->out(this->_tensor);
            }
        }
    };

    //
    class UnaryOPNode final : public tff::core::graph::GraphNode {
    public:
        explicit UnaryOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_UNARY);
            this->_builder = std::make_shared<kernel::UnaryOPBuilder>();
        }

        ~UnaryOPNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_UNARY) {
                tff::log::Logger::error("UnaryOPNode op type(expect TFF_OP_UNARY) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::UnaryOPBuilder>(this->_builder);
            auto x1 = builder->x1<std::shared_ptr<core::memory::Tensor> >();
            auto x2 = builder->x2<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::UnaryOPBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect UnaryOPBuilder) is wrong!!");
                return;
            }
            auto x1 = builder->x1<std::shared_ptr<core::memory::Tensor> >();
            auto x2 = builder->x2<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(x1, kernel::UnaryOPBuilder::Params::X1));
            this->_tensor_param_map.insert(std::make_pair(x2, kernel::UnaryOPBuilder::Params::X2));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<tff::core::memory::Tensor>(x1);
                builder->out(this->_tensor);
            }
        }
    };

    //
    class BinaryOPNode final : public tff::core::graph::GraphNode {
    public:
        explicit BinaryOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_BINARY);
            this->_builder = std::make_shared<kernel::BinaryOPBuilder>();
        }

        ~BinaryOPNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_BINARY) {
                tff::log::Logger::error("BinaryOPNode op type(expect TFF_OP_BINARY) is wrong!!");
                return;
            }
            if (this->input_nodes().size() != 2) {
                tff::log::Logger::error("BinaryOPNode input node size(expect 2) is wrong!!");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::BinaryOPBuilder>(this->_builder);
            auto x1 = builder->x1<std::shared_ptr<core::memory::Tensor> >();
            auto x2 = builder->x2<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::BinaryOPBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect BinaryOPBuilder) is wrong!!");
                return;
            }
            auto x1 = builder->x1<std::shared_ptr<core::memory::Tensor> >();
            auto x2 = builder->x2<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(x1, kernel::BinaryOPBuilder::Params::X1));
            this->_tensor_param_map.insert(std::make_pair(x2, kernel::BinaryOPBuilder::Params::X2));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<tff::core::memory::Tensor>(x1);
                builder->out(this->_tensor);
            }
        }
    };

    //
    class MaskOPNode final : public tff::core::graph::GraphNode {
    public:
        explicit MaskOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_ATTN_MASK);
            this->_builder = std::make_shared<kernel::MaskOPBuilder>();
        }

        ~MaskOPNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_ATTN_MASK) {
                tff::log::Logger::error("MaskOPNode op type(expect TFF_OP_ATTN_MASK) is wrong!!");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::MaskOPBuilder>(this->_builder);
            auto data_type = static_cast<memory::DataType>(builder->data_type<memory::DataType>());
            auto token_num = builder->token_num<const int>();
            auto input_tensor = builder->in<std::shared_ptr<memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::MaskOPBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MaskOPBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::MaskOPBuilder::Params::In));
            auto data_type = (builder->data_type<memory::DataType>());
            auto token_num = builder->token_num<const int>();
            if (this->_tensor == nullptr) {
                this->_tensor = input_tensor;
                builder->out(this->_tensor);
            }
        }
    };

    //
    class GetOfRowsOPNode final : public tff::core::graph::GraphNode {
    public:
        explicit GetOfRowsOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_GET_ROWS);
            this->_builder = std::make_shared<kernel::GetRowBuilder>();
        }

        ~GetOfRowsOPNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_GET_ROWS) {
                tff::log::Logger::error("GetOfRowsOPNode op type (expect TFF_OP_GET_ROWS) is wrong");
                return;
            }
            if (this->input_nodes().size() != 1) {
                tff::log::Logger::error("GetOfRowsOPNode input_nodes size (expect 1)");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::GetRowBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            auto seq_id = builder->seq_id<const int>();
            auto layer_id = builder->layer_id<const int>();
            auto kv_cache_ctx =
                    builder->kv_cache_ctx<std::shared_ptr<runtime::LLMKVCache> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::GetRowBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect GetRowBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::GetRowBuilder::Params::In));
            auto seq_id = builder->seq_id<int>();
            auto layer_id = builder->layer_id<int>();
            auto kv_cache_ctx =
                    builder->kv_cache_ctx<std::shared_ptr<runtime::LLMKVCache> >();
            auto max_seq_len = builder->max_seq_len<int>();
            std::array<int64_t, MAX_TENSOR_DIM> shape = {
                1, input_tensor->get_shape()[1], max_seq_len, input_tensor->get_shape()[3]
            };
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(memory::DataType::TFF_DATA_TYPE_I64,
                                                                 memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                 shape);
                this->_tensor->set_tensor_type(input_tensor->get_tensor_type());
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class SetOfRowsOPNode final : public tff::core::graph::GraphNode {
    public:
        explicit SetOfRowsOPNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_SET_ROWS);
            this->_builder = std::make_shared<kernel::SetRowBuilder>();
        }

        ~SetOfRowsOPNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_SET_ROWS) {
                tff::log::Logger::error("SetOfRowsOPNode op type (expect TFF_OP_SET_ROWS) is wrong");
                return;
            }
            if (this->input_nodes().size() != 1) {
                tff::log::Logger::error("SetOfRowsOPNode input_nodes size (expect 1)");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::SetRowBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            auto seq_id = builder->seq_id<int>();
            auto layer_id = builder->layer_id<int>();
            auto kv_cache_ctx =
                    builder->kv_cache_ctx<std::shared_ptr<runtime::LLMKVCache> >();

            const auto dim0 = input_tensor->get_shape()[0];
            const auto dim1 = input_tensor->get_shape()[1];
            const auto dim2 = input_tensor->get_shape()[2];
            const auto batch = input_tensor->get_shape()[3];

            const int total_token_num = kv_cache_ctx->get_kv_token_num(seq_id, layer_id);
            const int pre_token_num = total_token_num - dim2;
            if (pre_token_num < 0) {
                log::Logger::error("invalid kv cache");
                return;
            }
            for (int i = 0; i < dim2; i += PAGE_SIZE) {
                const auto &[page_id, offset] =
                        kv_cache_ctx->get_location(seq_id, layer_id, pre_token_num + i);

                if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K_NORM) {
                    auto cache_tensor = kv_cache_ctx->get_k(seq_id, layer_id, page_id, this->event());
                    if (cache_tensor == nullptr) {
                        tff::log::Logger::error("k cache is invalid");
                        continue;
                    }
                    if (cache_tensor->get_buffer() == nullptr) {
                        tff::log::Logger::error("k cache buffer is invalid");
                        continue;
                    }
                } else if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
                    auto cache_tensor = kv_cache_ctx->get_v(seq_id, layer_id, page_id, this->event());
                    if (cache_tensor == nullptr) {
                        tff::log::Logger::error("v cache is invalid");
                        continue;
                    }
                    if (cache_tensor->get_buffer() == nullptr) {
                        tff::log::Logger::error("v cache buffer is invalid");
                        continue;
                    }
                }
            }
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::SetRowBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect SetRowBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::SetRowBuilder::Params::In));
            auto seq_id = builder->seq_id<const int>();
            auto layer_id = builder->layer_id<const int>();
            auto kv_cache_ctx =
                    builder->kv_cache_ctx<std::shared_ptr<runtime::LLMKVCache> >();
            auto data_type = builder->data_type<memory::DataType>();
            auto tensor_type = builder->tensor_type<memory::ModelTensorType>();
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(data_type,
                                                                 memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                 input_tensor->get_shape());
                this->_tensor->set_tensor_type(tensor_type);
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class QuantNode final : public tff::core::graph::GraphNode {
    public:
        explicit QuantNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_QUANTIZE);
            this->_builder = std::make_shared<kernel::QuantBuilder>();
        }

        ~QuantNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_QUANTIZE) {
                tff::log::Logger::error("QuantQ80Node op type (expect TFF_OP_QUANTIZE_Q8) is wrong");
                return;
            }
            if (this->_input_nodes.size() != 1) {
                tff::log::Logger::error("QuantQ80Node op type (expect 1) is wrong");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::QuantBuilder>(this->_builder);
            auto quant_data_type = (builder->quant_data_type<memory::DataType>());
            const auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::QuantBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect QuantBuilder) is wrong!!");
                return;
            }
            auto quant_data_type = (builder->quant_data_type<memory::DataType>());
            const auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::QuantBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(quant_data_type,
                                                                 memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                 input_tensor->get_shape());
                this->_tensor->set_tensor_type(input_tensor->get_tensor_type());
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class QuantAlignedNode final : public tff::core::graph::GraphNode {
    public:
        explicit QuantAlignedNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_QUANTIZE_ALIGNED);
            this->_builder = std::make_shared<kernel::QuantAlignedBuilder>();
        }

        ~QuantAlignedNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_QUANTIZE_ALIGNED) {
                tff::log::Logger::error("QuantAlignedNode op type (expect TFF_OP_QUANTIZE_ALIGNED) is wrong");
                return;
            }
            if (this->_input_nodes.size() != 1) {
                tff::log::Logger::error("QuantAlignedNode op type (expect 1) is wrong");
                return;
            }


            auto builder =
                    std::dynamic_pointer_cast<kernel::QuantAlignedBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::QuantAlignedBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect QuantAlignedBuilder) is wrong!!");
                return;
            }
            auto quant_data_type = (builder->quant_data_type<memory::DataType>());
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::QuantAlignedBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(quant_data_type,
                                                                 input_tensor->memory_type(),
                                                                 input_tensor->get_shape());
                input_tensor->set_memory_type(memory::MemoryType::TFF_MEM_TYPE_WORKSPACE);
                this->_tensor->set_tensor_type(input_tensor->get_tensor_type());
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    class ViewNode final : public tff::core::graph::GraphNode {
    public:
        explicit ViewNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_VIEW);
            this->_builder = std::make_shared<kernel::ViewOPBuilder>();
        }

        ~ViewNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_VIEW) {
                tff::log::Logger::error("");
                return;
            }
            if (this->input_nodes().size() != 1) {
                tff::log::Logger::error("ViewNode input is invalid");
                return;
            }

            auto builder = std::dynamic_pointer_cast<kernel::ViewOPBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::ViewOPBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect ViewOPBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::ViewOPBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = input_tensor;
                builder->out(this->_tensor);
            }
        }
    };

    //
    class MemOptNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemOptNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MEM_RECYCLE);
            this->_builder = std::make_shared<kernel::MemOptOPBuilder>();
        }

        ~MemOptNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_MEM_RECYCLE) {
                tff::log::Logger::error("");
                return;
            }
            auto device_id = this->_tensor->get_allocator()->_device_id;
            this->_mem_manager_ptr->reclaim_memory(device_id);
            auto builder = std::dynamic_pointer_cast<kernel::MemOptOPBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::MemOptOPBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect MemOptOPBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::MemOptOPBuilder::Params::In));
            if (this->_tensor == nullptr) {
                this->_tensor = input_tensor;
                builder->out(this->_tensor);
            }
        }
    };

    //
    class ConvertNode final : public tff::core::graph::GraphNode {
    public:
        explicit ConvertNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_CONVERT);
            this->_builder = std::make_shared<kernel::ConvertOPBuilder>();
        }

        ~ConvertNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_CONVERT) {
                tff::log::Logger::error("");
                return;
            }
            if (this->input_nodes().size() != 1) {
                tff::log::Logger::error("ConvertNode input is invald");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::ConvertOPBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            auto data_type = static_cast<memory::DataType>(builder->convert_data_type<memory::DataType>());
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::ConvertOPBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect ConvertOPBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::ConvertOPBuilder::Params::In));
            auto data_type = static_cast<memory::DataType>(builder->convert_data_type<memory::DataType>());
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<core::memory::Tensor>(data_type,
                                                                       memory::MemoryType::TFF_MEM_TYPE_WORKSPACE,
                                                                       input_tensor->get_shape());
                this->_tensor->set_tensor_type(input_tensor->get_tensor_type());
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };

    //
    //
    class GatherNode final : public tff::core::graph::GraphNode {
    public:
        explicit GatherNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_GATHER);
            this->_builder = std::make_shared<kernel::GatherOPBuilder>();
        }

        ~GatherNode() override = default;

    public:
        void prepare_params() override {
            if (this->_op_type != TFF_OP_GATHER) {
                tff::log::Logger::error("");
                return;
            }
            if (this->input_nodes().size() != 1) {
                tff::log::Logger::error("GatherNode input is invalid");
                return;
            }
            auto builder = std::dynamic_pointer_cast<kernel::GatherOPBuilder>(this->_builder);
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            auto row_index = builder->row_index<const std::vector<int>>();
        }

        void shape_infer() {
            this->_tensor_param_map.clear();
            auto builder = std::dynamic_pointer_cast<kernel::GatherOPBuilder>(this->_builder);
            if (builder == nullptr) {
                tff::log::Logger::error("MatMulNode prepare_params op type(expect GatherOPBuilder) is wrong!!");
                return;
            }
            auto input_tensor = builder->in<std::shared_ptr<core::memory::Tensor> >();
            this->_tensor_param_map.insert(std::make_pair(input_tensor, kernel::GatherOPBuilder::Params::In));
            auto row_index = builder->row_index<const std::vector<int>>();
            std::array<int64_t, MAX_TENSOR_DIM> shape = {
                input_tensor->get_shape()[0], static_cast<int64_t>(row_index.size()),
                input_tensor->get_shape()[2], input_tensor->get_shape()[3]
            };
            if (this->_tensor == nullptr) {
                this->_tensor = std::make_shared<memory::Tensor>(input_tensor->get_data_type(),
                                                                 memory::MemoryType::TFF_MEM_TYPE_WORKSPACE, shape);
                this->_tensor->set_tensor_type(input_tensor->get_tensor_type());
                this->_tensor->set_allocator(input_tensor->get_allocator());
                builder->out(this->_tensor);
            }
        }
    };
}


#endif //TFFINFER_TFFOPNODE_H
