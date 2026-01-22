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
    void unary_op(const core::graph::TFFUnaryType &type,
                  std::shared_ptr<core::memory::Tensor> &x1, std::shared_ptr<core::memory::Tensor> &x2,
                  std::shared_ptr<core::memory::Tensor> &result,
                             std::shared_ptr<core::device::DeviceStream> &stream,
                             std::shared_ptr<core::device::DeviceEvent> &event,
                             std::vector<std::shared_ptr<core::device::DeviceEvent>> &wait_event_list) {
        if (x1 == nullptr || x2 == nullptr || result == nullptr) {
            tff::log::Logger::error("input params is invalid!!");
            return;
        }
        const int64_t n1 = x1->get_shape()[0];
        const int64_t n2 = x2->get_shape()[0];
        const int64_t k = x1->get_shape()[0] * x1->get_shape()[1] * x1->get_shape()[2] * x1->get_shape()[3];
        for (auto &wait_event : wait_event_list) {
            stream->wait_event(wait_event->get_native_event());
        }
        switch (type) {
            case core::graph::TFFUnaryType::TFF_UNARY_TYPE_SILU: {
                constexpr int BLOCK_SIZE = 256;
                dim3 grid((k + BLOCK_SIZE - 1) / BLOCK_SIZE);
                dim3 block(BLOCK_SIZE);

                op_unary<T, op_silu><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>>(static_cast<T *>(x1->get_buffer()->ptr()),
                                                      static_cast<T *>(x2->get_buffer()->ptr()),
                                                      static_cast<T *>(result->get_buffer()->ptr()),
                                                      k, n1, n2);
                event->record(stream);
                break;
            }

            default:
                break;
        }
    }

    //
    template<typename T>
    void tff::kernel::UnaryOP<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), UnaryOP<T>::get_op_name().c_str());
        auto unary_type = static_cast<tff::core::graph::TFFUnaryType>(get_param_value<const int>(1, para_ptr));
        auto x1 = get_param_value<std::shared_ptr<core::memory::Tensor> >(2, para_ptr);
        auto x2 = get_param_value<std::shared_ptr<core::memory::Tensor> >(3, para_ptr);
        auto output = get_param_value<std::shared_ptr<core::memory::Tensor> >(4, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
                    std::shared_ptr<
                        tff::core::runtime::LLMMemManager> >(5, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream>>(6, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent>>(7, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent>>>(8, para_ptr);
        if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }

        unary_op<T>(unary_type, x1, x2, output, stream, event, event_list);
    }

    template<typename T>
    std::string tff::kernel::UnaryOP<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_UNARY);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;

        return name;
    }


    template class tff::kernel::UnaryOP<half>;
    template class tff::kernel::UnaryOP<float>;
    REGISTER_OP_OBJECT(UnaryOP, half);

    REGISTER_OP_OBJECT(UnaryOP, float);
}
