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
    class MatMulNode : public tff::core::graph::GraphNode {
    public:
        explicit MatMulNode(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_MUL_MAT);
        }

        ~MatMulNode() override = default;

    public:
        using callback_para = void(
            tff::kernel::base::Layout,
            tff::kernel::base::Transpose,
            tff::kernel::base::Transpose,
            size_t, size_t, size_t,
            float,
            tff::kernel::Buffer<float> &, size_t, size_t,
            tff::kernel::Buffer<float> &, size_t, size_t,
            float,
            tff::kernel::Buffer<float> &, size_t, size_t);

        template<typename T>
        auto get_xgemm_callback() {
            return tff::factory::FunctionFactory::instance()->get_callback<callback_para>(
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
            using CallBack = std::function<callback_para>;
            CallBack callback;
            switch (A->get_data_type()) {
                case memory::TFF_DATA_TYPE_F32: {
                    callback = get_xgemm_callback<float>();
                    break;
                }

                case memory::TFF_DATA_TYPE_F64: {
                    callback = get_xgemm_callback<double>();
                    break;
                }
                default:
                    throw std::runtime_error("Unsupported data type");
            }
            //
            if (callback) {
                const auto &shape_a = A->get_shape();
                const auto &shape_b = B->get_shape();
                int64_t m = shape_a[shape_a.size() - 2];
                int64_t k1 = shape_a[shape_a.size() - 1];
                int64_t k2 = shape_b[shape_b.size() - 2];
                int64_t n = shape_b[shape_b.size() - 1];
                auto a_buffer = tff::kernel::Buffer<float>((float *) (A->get_buffer()->ptr()), A->get_allocator());
                auto b_buffer = tff::kernel::Buffer<float>((float *) (B->get_buffer()->ptr()), B->get_allocator());
                auto c_buffer = tff::kernel::Buffer<float>((float *) (C->get_buffer()->ptr()), C->get_allocator());
                callback(tff::kernel::base::Layout::kColMajor,
                         tff::kernel::base::Transpose::kNo,
                         tff::kernel::base::Transpose::kNo,
                         m, n, k1,
                         0,
                         a_buffer, 0, m,
                         b_buffer, 0, n,
                         0,
                         c_buffer, 0, m);
            }
            return true;
        }
    };
    //
    class UploadBuffer : public tff::core::graph::GraphNode {
    public:
        explicit UploadBuffer(const std::string &name = "") : GraphNode(name) {
            set_op_type(TFF_OP_GET_ROWS);
        }
         ~UploadBuffer() override = default;

    public:
        using callback_para = void();

        template<typename T>
        auto get_read_buffer_callback() {
            return tff::factory::FunctionFactory::instance()->get_callback<callback_para>(
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
            }else {
                if (this->_src_tensors_ptr[0]->get_allocator()->device_type() == tff::core::device::DeviceType::TFF_BACKEND_DEVICE_TYPE_GPU) {
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
}


#endif //TFFINFER_TFFOPNODE_H
