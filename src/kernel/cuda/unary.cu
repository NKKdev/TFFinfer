//
// Created by nkk on 2026/1/21.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"


namespace tff::kernel {
    static __device__ __host__ __forceinline__ float op_silu(float x) {
        return x / (1.0f + expf(-x));
    }

    template<typename T, float (*op)(float)>
    __global__ __forceinline__ void op_unary(const T *__restrict__ x1, const T *__restrict__ x2,
                                             T *result, const int64_t k, const int64_t n1, const int64_t n2) {
        const int64_t g_index = blockDim.x * blockIdx.x + threadIdx.x;
        if (g_index >= k) {
            return;
        }
        const int64_t row = g_index / n1;
        const int64_t col = g_index % n2;
        const int64_t index1 = row * n1 + col;
        const int64_t index2 = row * n2 + col;

        result[g_index] = (T) op(x1[index1]) * x2[index2];
    }

    template<typename T>
    class UnaryOP<T, core::device::GPUTag> : public base::OPCreatorBase<UnaryOP<T, core::device::GPUTag>, T, core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_UNARY;
        }
    };
    template<typename T>
    void unary_op(const core::graph::TFFUnaryType &type,
                  std::shared_ptr<core::memory::Tensor> &x1, std::shared_ptr<core::memory::Tensor> &x2,
                  std::shared_ptr<core::memory::Tensor> &result,
                             std::shared_ptr<core::device::DeviceStream> &stream) {
        if (x1 == nullptr || x2 == nullptr || result == nullptr || x1->get_buffer() == nullptr ||
            x2->get_buffer() == nullptr || result->get_buffer() == nullptr) {
            tff::log::Logger::error("input params is invalid!!");
            return;
        }
        const int64_t n1 = x1->get_shape()[0];
        const int64_t n2 = x2->get_shape()[0];
        const int64_t k = x1->get_shape()[0] * x1->get_shape()[1] * x1->get_shape()[2] * x1->get_shape()[3];
        switch (type) {
            case core::graph::TFFUnaryType::TFF_UNARY_TYPE_SILU: {
                constexpr int BLOCK_SIZE = 256;
                dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE);
                dim3 block(BLOCK_SIZE);

                op_unary<T, op_silu><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>>(static_cast<T *>(x1->get_buffer()->ptr()),
                                                      static_cast<T *>(x2->get_buffer()->ptr()),
                                                      static_cast<T *>(result->get_buffer()->ptr()),
                                                      k, n1, n2);
                break;
            }

            default:
                break;
        }
    }

    //
    template<typename T>
    void tff::kernel::UnaryOP<T, core::device::GPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {

        auto unary_type = static_cast<tff::core::graph::TFFUnaryType>(kernel::base::get_param_value<int>(
            UnaryOPBuilder::Params::UnaryType, para_ptr));
        auto x1 = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(UnaryOPBuilder::Params::X1, para_ptr);
        auto x2 = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(UnaryOPBuilder::Params::X2, para_ptr);
        auto output = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(UnaryOPBuilder::Params::Out, para_ptr);
        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        kernel::builder::OpParamBuilderBase<UnaryOPBuilder>::CommonParams::Stream, para_ptr);
        unary_op<T>(unary_type, x1, x2, output, stream);
    }

    template class tff::kernel::UnaryOP<half, core::device::GPUTag>;
    template class tff::kernel::UnaryOP<float, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(UnaryOP, half, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(UnaryOP, float, core::device::GPUTag);
}
