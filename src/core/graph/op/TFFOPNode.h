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
                CREATE_LAYER_FLAG, tff::kernel::XGemm<T>::get_op_name());
        }

    public:
        bool init() override {
            // 调用基类 init 做通用检查
            if (!GraphNode::init()) {
                return false;
            }

            // MatMul 特有检查
            if (_src_tensors_ptr.size() != 2) {
                tff::log::Logger::error("[MatMulNode]: %s: requires 2 inputs, got :%d input!!\n", _name.c_str(),
                                        this->_src_tensors_ptr.size());
                return false;
            }

            auto A = _src_tensors_ptr[0];
            auto B = _src_tensors_ptr[1];
            auto C = std::make_shared<tff::core::memory::Tensor>();

            const auto &shape_a = A->get_shape();
            const auto &shape_b = B->get_shape();

            if (shape_a.size() < 2 || shape_b.size() < 2) {
                tff::log::Logger::error("[MatMulNode]: %s: inputs must be at least 2D!!\n", _name.c_str());
                return false;
            }

            int64_t m = shape_a[shape_a.size() - 2];
            int64_t k1 = shape_a[shape_a.size() - 1];
            int64_t k2 = shape_b[shape_b.size() - 2];
            int64_t n = shape_b[shape_b.size() - 1];

            if (k1 != k2) {
                tff::log::Logger::error("[MatMulNode]: shape mismatch");
                return false;
            }

            // 设置输出 shape: [..., m, n]
            auto out_shape = std::vector<int64_t>(shape_a.begin(), shape_a.end() - 1);
            out_shape.push_back(n);
            C->set_shape(out_shape);
            C->set_data_type(A->get_data_type()); // 输出类型与输入一致
            C->set_allocator(A->get_allocator());
            C->allocate();
            this->_dst_tensors_ptr.push_back(std::move(C));
            return true;
        }

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
                CREATE_LAYER_FLAG, tff::kernel::XGemm<T>::get_op_name());
        }

    public:
        bool init() override {
            // 调用基类 init 做通用检查
            if (!GraphNode::init()) {
                return false;
            }

            // MatMul 特有检查
            if (_src_tensors_ptr.empty()) {
                tff::log::Logger::error("[MatMulNode]: %s: requires 1 inputs, got :%d input!!\n", _name.c_str(),
                                        this->_src_tensors_ptr.size());
                return false;
            } else {
                if (this->_src_tensors_ptr[0]->get_allocator()->device_type() ==
                    tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU) {
                    tff::log::Logger::error("[MatMulNode]: %s: requires 1 inputs, got :%d input!!\n", _name.c_str(),
                                            this->_src_tensors_ptr.size());
                    return false;
                }
            }

            auto A = _src_tensors_ptr[0];
            auto B = _src_tensors_ptr[1];

            const auto &shape_a = A->get_shape();
            const auto &shape_b = B->get_shape();
            //todo
            return true;
        }

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
        bool init() override {
            if (!GraphNode::init()) {
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
        bool init() override {
            if (!GraphNode::init()) {
                return false;
            }
            // RMSNorm 通常需要 2 个输入：输入 X 和权重 Weight
            if (_src_tensors_ptr.size() != 2) {
                tff::log::Logger::error("[RMSNormNode]: %s: requires 2 inputs (X, Weight), got %d input(s)!!\n",
                                        _name.c_str(), static_cast<int>(_src_tensors_ptr.size()));
                return false;
            }
            auto X = _src_tensors_ptr[0];
            auto Weight = _src_tensors_ptr[1];

            // 检查 Weight 形状是否与 X 的最后一个维度匹配
            const auto &shape_x = X->get_shape();
            const auto &shape_weight = Weight->get_shape();
            if (shape_weight.size() != 1 || shape_weight[0] != shape_x.back()) {
                tff::log::Logger::error("[RMSNormNode]: %s: Weight shape mismatch. Expected [last_dim_of_X], got %d.\n",
                                        _name.c_str(), shape_weight[0]);
                return false;
            }

            // 输出形状通常与输入 X 相同
            auto Y = std::make_shared<tff::core::memory::Tensor>();
            Y->set_shape(shape_x);
            Y->set_data_type(X->get_data_type());
            Y->set_allocator(X->get_allocator());
            Y->allocate();
            this->_dst_tensors_ptr.clear();
            this->_dst_tensors_ptr.push_back(std::move(Y));
            return true;
        }

        bool forward() override {
            if (!GraphNode::forward()) {
                tff::log::Logger::error("[RMSNormNode]: forward() failed!!\n");
                return false;
            }

            if (_src_tensors_ptr.size() != 2 || _dst_tensors_ptr.size() != 1) {
                tff::log::Logger::error("[RMSNormNode]: %s: Incorrect number of inputs/outputs in forward.\n",
                                        _name.c_str());
                return false;
            }

            auto X = _src_tensors_ptr[0].get();
            auto Weight = _src_tensors_ptr[1].get();
            auto Y = _dst_tensors_ptr[0].get();

            // --- Kernel 调用 ---
            tff::log::Logger::info("[RMSNormNode]: %s: forward logic not implemented.\n", _name.c_str());
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
        bool init() override {
            if (!_src_tensors_ptr.empty()) {
                auto output = std::make_shared<tff::core::memory::Tensor>();

                output->set_shape(_src_tensors_ptr[0]->get_shape());

                output->set_data_type(_src_tensors_ptr[0]->get_data_type());

                output->set_allocator(_src_tensors_ptr[0]->get_allocator());

                output->allocate();

                _dst_tensors_ptr.push_back(std::move(output));
            }

            return true;
        }


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

        bool init() override {
            if (_src_tensors_ptr.size() != 1) {
                tff::log::Logger::error("[DupNode]: %s: requires 1 input, got %d inputs\n",
                                        _name.c_str(), static_cast<int>(_src_tensors_ptr.size()));
                return false;
            }
            auto input = _src_tensors_ptr[0];
            auto output = std::make_shared<tff::core::memory::Tensor>();
            output->set_shape(input->get_shape());
            output->set_data_type(input->get_data_type());
            output->set_allocator(input->get_allocator());
            output->allocate();
            _dst_tensors_ptr.push_back(std::move(output));
            return true;
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
        bool init() override {
            if (_src_tensors_ptr.size() != 1) {
                tff::log::Logger::error("[SqrNode]: %s: requires 1 input, got %d inputs\n",

                                        _name.c_str(), static_cast<int>(_src_tensors_ptr.size()));
                return false;
            }


            auto input = _src_tensors_ptr[0];
            auto output = std::make_shared<tff::core::memory::Tensor>();
            output->set_shape(input->get_shape());
            output->set_data_type(input->get_data_type());
            output->set_allocator(input->get_allocator());
            output->allocate();
            _dst_tensors_ptr.push_back(std::move(output));
            return true;
        }

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
            //     tff::log::Logger::error("[SqrNode]: %s: failed to get sqr kernel\n", _name.c_str());
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
        bool init() override {
            if (_src_tensors_ptr.size() != 1) {
                tff::log::Logger::error("[SqrtNode]: %s: requires 1 input, got %d inputs\n",

                                        _name.c_str(), static_cast<int>(_src_tensors_ptr.size()));
                return false;
            }


            auto input = _src_tensors_ptr[0];
            auto output = std::make_shared<tff::core::memory::Tensor>();
            output->set_shape(input->get_shape());
            output->set_data_type(input->get_data_type());
            output->set_allocator(input->get_allocator());
            output->allocate();
            _dst_tensors_ptr.push_back(std::move(output));
            return true;
        }

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
            //     tff::log::Logger::error("[SqrtNode]: %s: failed to get sqrt kernel\n", _name.c_str());
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
        bool init() override {
            if (!GraphNode::init()) return false;
            if (_src_tensors_ptr.size() != 2) {
                /* log error */
                return false;
            }
            return true;
        }

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

        bool init() override {
            /* ... */
        }

        bool forward() override {
            /* ... */
        }
    };

    //
    class DivNode final : public tff::core::graph::GraphNode {
    public:
        explicit DivNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_DIV); }

        ~DivNode() override = default;

        bool init() override {
            /* ... */
        }

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
        bool init() override {
            if (!GraphNode::init()) return false;
            if (_src_tensors_ptr.size() < 1 || _src_tensors_ptr.size() > 2) {
                /* log error */
                return false;
            }
            return true;
        }

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
        bool init() override {
            if (!GraphNode::init()) return false;
            if (_src_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }

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
        bool init() override {
            if (!GraphNode::init()) return false;

            if (_src_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }

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
        bool init() override {
            if (!GraphNode::init()) return false;

            if (_src_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }

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
        bool init() override {
            if (!GraphNode::init()) return false;

            if (_src_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }

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
    class MemCpyNode final : public tff::core::graph::GraphNode {
    public:
        explicit MemCpyNode(const std::string &name = "") : GraphNode(name) { set_op_type(TFF_OP_MEM_CPY); }

        ~MemCpyNode() override = default;

    public:
        bool init() override {
            if (!GraphNode::init()) return false;

            if (_src_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }

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
        bool init() override {
            if (!GraphNode::init()) return false;

            if (_src_tensors_ptr.size() != 1) {
                /* log error */
                return false;
            }
            return true;
        }

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
