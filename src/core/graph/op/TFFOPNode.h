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

        bool forward() override {
            if (!GraphNode::forward()) {
                tff::log::Logger::error("[MatMulNode]: forward() failed!!\n");
                return false;
            }

            if (_src_tensors_ptr.size() != 2 || _dst_tensors_ptr.size() != 1) {
                return false;
            }

            auto A = _src_tensors_ptr[0].get();
            auto B = _src_tensors_ptr[1].get();
            auto C = _dst_tensors_ptr[0].get();

            //kernel compute
            // using CallBack = std::function<callback_para>;
            // CallBack callback;
            // switch (A->get_data_type()) {
            //     case memory::TFF_DATA_TYPE_F32: {
            //         callback = get_xgemm_callback<float>();
            //         break;
            //     }
            //
            //     case memory::TFF_DATA_TYPE_F64: {
            //         callback = get_xgemm_callback<double>();
            //         break;
            //     }
            //     default:
            //         throw std::runtime_error("Unsupported data type");
            // }
            // //
            // if (callback) {
            //     const auto &shape_a = A->get_shape();
            //     const auto &shape_b = B->get_shape();
            //     int64_t m = shape_a[shape_a.size() - 2];
            //     int64_t k1 = shape_a[shape_a.size() - 1];
            //     int64_t k2 = shape_b[shape_b.size() - 2];
            //     int64_t n = shape_b[shape_b.size() - 1];
            //     auto a_buffer = tff::kernel::Buffer<float>((float *) (A->get_buffer()->ptr()), A->get_allocator());
            //     auto b_buffer = tff::kernel::Buffer<float>((float *) (B->get_buffer()->ptr()), B->get_allocator());
            //     auto c_buffer = tff::kernel::Buffer<float>((float *) (C->get_buffer()->ptr()), C->get_allocator());
            //     callback(tff::kernel::base::Layout::kColMajor,
            //              tff::kernel::base::Transpose::kNo,
            //              tff::kernel::base::Transpose::kNo,
            //              m, n, k1,
            //              0,
            //              a_buffer, 0, m,
            //              b_buffer, 0, n,
            //              0,
            //              c_buffer, 0, m);
            // }
            return true;
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

        bool forward() override {
            if (!GraphNode::forward()) {
                tff::log::Logger::error("[MatMulNode]: forward() failed!!\n");
                return false;
            }

            if (_src_tensors_ptr.size() != 2 || _dst_tensors_ptr.size() != 1) {
                return false;
            }

            auto A = _src_tensors_ptr[0].get();
            auto B = _src_tensors_ptr[1].get();
            auto C = _dst_tensors_ptr[0].get();

            //kernel compute todo

            return true;
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
        bool forward() override {
            if (!GraphNode::forward()) {
                tff::log::Logger::error("[AddNode]: forward() failed!!\n");
                return false;
            }
            if (_src_tensors_ptr.size() != 2 || _dst_tensors_ptr.size() != 1) {
                return false;
            }
            auto A = _src_tensors_ptr[0].get();
            auto B = _src_tensors_ptr[1].get();
            auto C = _dst_tensors_ptr[0].get();
            return true;
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

        bool forward() override {
            if (!GraphNode::forward()) {
                tff::log::Logger::error("[RMSNormNode]: forward() failed!!\n");
                return false;
            }

            if (_src_tensors_ptr.size() != 2 || _dst_tensors_ptr.size() != 1) {
                tff::log::Logger::error("[RMSNormNode]: %s: Incorrect number of inputs/outputs in forward.\n",
                                        this->_node_metadata._name.c_str());
                return false;
            }

            auto X = _src_tensors_ptr[0].get();
            auto Weight = _src_tensors_ptr[1].get();
            auto Y = _dst_tensors_ptr[0].get();

            // --- Kernel 调用 ---
            tff::log::Logger::info("[RMSNormNode]: %s: forward logic not implemented.\n",
                                   this->_node_metadata._name.c_str());
            return true;
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

        bool forward() override {
            if (_src_tensors_ptr.empty() || _dst_tensors_ptr.empty()) {
                return true;
            }
            auto src = _src_tensors_ptr[0].get();
            auto dst = _dst_tensors_ptr[0].get();
            return true;
        }
    };

    //
    class DupNode final : public tff::core::graph::GraphNode {
    public:
        explicit DupNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_DUP);
        }


        bool forward() override {
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                return false;
            }

            auto src = _src_tensors_ptr[0].get();
            auto dst = _dst_tensors_ptr[0].get();
            return true;
        }
    };

    //
    class SqrNode final : public tff::core::graph::GraphNode {
    public:
        explicit SqrNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_SQR);
        }

    public:
        bool forward() override {
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                return false;
            }

            auto input = _src_tensors_ptr[0].get();
            auto output = _dst_tensors_ptr[0].get();

            // auto callback = tff::factory::FunctionFactory::instance()->get_callback(
            //     CREATE_LAYER_FLAG, "sqr");
            //
            // if (!callback) {
            //     tff::log::Logger::error("[SqrNode]: %s: failed to get sqr kernel\n", this->_node_metadata._name.c_str());
            //
            //     return false;
            // }

            auto in_buffer = tff::kernel::Buffer<float>((float *) (input->get_buffer()->ptr()), input->get_allocator());

            auto out_buffer = tff::kernel::Buffer<float>((float *) (output->get_buffer()->ptr()),
                                                         output->get_allocator());
            return true;
        }
    };

    //
    class SqrtNode final : public tff::core::graph::GraphNode {
    public:
        explicit SqrtNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_SQRT);
        }

    public:

        bool forward() override {
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                return false;
            }
            auto input = _src_tensors_ptr[0].get();
            auto output = _dst_tensors_ptr[0].get();

            using callback_para = void(size_t, const float *, float *);

            // auto callback = tff::factory::FunctionFactory::instance()->get_callback(
            //
            //     CREATE_LAYER_FLAG, "sqrt");
            //
            // if (!callback) {
            //     tff::log::Logger::error("[SqrtNode]: %s: failed to get sqrt kernel\n", this->_node_metadata._name.c_str());
            //
            //     return false;
            // }


            auto in_buffer = tff::kernel::Buffer<float>((float *) (input->get_buffer()->ptr()), input->get_allocator());

            auto out_buffer = tff::kernel::Buffer<float>((float *) (output->get_buffer()->ptr()),
                                                         output->get_allocator());

            return true;
        }
    };

    //
    class SubNode final : public tff::core::graph::GraphNode {
    public:
        explicit SubNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_SUB); }

        ~SubNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }
            if (_src_tensors_ptr.size() != 2 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }

            return true;
        }
    };

    //
    class MulNode final : public tff::core::graph::GraphNode {
    public:
        explicit MulNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MUL); }

        ~MulNode() override = default;

        bool forward() override {
            /* ... */
        }
    };

    //
    class DivNode final : public tff::core::graph::GraphNode {
    public:
        explicit DivNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_DIV); }

        ~DivNode() override = default;

        bool forward() override {
            /* ... */
        }
    };

    //
    class ReshapeNode final : public tff::core::graph::GraphNode {
    public:
        explicit ReshapeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_RESHAPE); }

        ~ReshapeNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }

            auto input = _src_tensors_ptr[0].get();

            auto output = _dst_tensors_ptr[0].get();

            return true;
        }
    };

    //
    class TransposeNode final : public tff::core::graph::GraphNode {
    public:
        explicit TransposeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_TRANSPOSE); }

        ~TransposeNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) { return false; }
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }
    };

    //
    class SoftmaxNode final : public tff::core::graph::GraphNode {
    public:
        explicit SoftmaxNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_SOFT_MAX); }

        ~SoftmaxNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }
    };

    //
    class RopeNode final : public tff::core::graph::GraphNode {
    public:
        explicit RopeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_ROPE); }

        ~RopeNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }
    };

    //
    class MapCPUBufferNode final : public tff::core::graph::GraphNode {
    public:
        explicit MapCPUBufferNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MAP2CPU); }

        ~MapCPUBufferNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) {
                return false;
            }
            if (this->_params_ptr->get_param_count() != 1) {
                tff::log::Logger::error("MapCPUBufferNode param count is %d(expect 1)",
                    this->_params_ptr->get_param_count());
                return false;
            }
            if (this->_op_type != TFF_OP_MAP2CPU) {
                tff::log::Logger::error("MapCPUBufferNode op type(expect TFF_OP_MAP2CPU) is wrong!!");
                return false;
            }
            auto callback = device()->get_op_func(this->_op_type);


            return true;
        }
    };

    //
    class MemCpyNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemCpyNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MEM_CPY); }

        ~MemCpyNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }
    };

    //
    class TokenizeNode final : public tff::core::graph::GraphNode {
    public:
        explicit TokenizeNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MEM_CPY); }

        ~TokenizeNode() override = default;

    public:

        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }
    };

    //
    class MemRefNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemRefNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_FLASH_ATTN_EXT); }

        ~MemRefNode() override = default;

    public:

        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }
    };

    //
    class FlashAttnNode final : public tff::core::graph::GraphNode {
    public:
        explicit FlashAttnNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_FLASH_ATTN_EXT); }

        ~FlashAttnNode() override = default;

    public:
        bool forward() override {
            if (!GraphNode::forward()) {
                /* log error */
                return false;
            }
            if (_src_tensors_ptr.size() != 1 || _dst_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }
    };
}


#endif //TFFINFER_TFFOPNODE_H
