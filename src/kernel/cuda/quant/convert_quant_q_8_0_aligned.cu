//
// Created by nkk on 2026/1/28.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T, const int WARP_SIZE, const int BLOCK_SIZE>
    __global__ __forceinline__ void quant_aligned_q_8_0(const T *__restrict__ src,
                                                        void *dst, const int M,
                                                        const int dst_stride_cnt) {
        const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = g_thread_id / WARP_SIZE;
        const int lane_id = g_thread_id % WARP_SIZE;

        const int row = blockIdx.y * blockDim.y + warp_id;
        const int col = blockIdx.x * blockDim.x + lane_id;
        const int start_dst_row = row;
        const int start_dst_col = col / BLOCK_SIZE;
        auto *dst_ptr = static_cast<tff::core::quant::Q_8_0_ALIGNED *>(dst);

        const int index = start_dst_row * dst_stride_cnt + start_dst_col;
        if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
            auto dst_val = &dst_ptr[index];
            auto src_val = &src[index];
            if (lane_id == 0) {
                dst_val->d = __half2float(src_val->d);
            }
            dst_val->qs[lane_id] = src_val->qs[lane_id];
        }
    }

    template<typename T>
    void quant_aligned(const int M, const int N,
                       std::shared_ptr<tff::core::memory::Tensor> &src,
                       std::shared_ptr<tff::core::memory::Tensor> &dst,
                       std::shared_ptr<core::device::DeviceStream> &stream,
                             std::shared_ptr<core::device::DeviceEvent> &event,
                             std::vector<std::shared_ptr<core::device::DeviceEvent>> &wait_event_list) {
        if constexpr (std::is_same_v<T, Q8_0>) {
            constexpr int BLOCK_SIZE = tff::core::quant::Q_8_0::BLOCK_SIZE;
            constexpr int VEC_M_DIM = 8;
            constexpr int WARP_NUM_PER_BLOCK = 8;
            for (auto &wait_event : wait_event_list) {
                if (wait_event == nullptr) {
                    continue;
                }
                stream->wait_event(wait_event->get_native_event());
            }
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
            dim3 block(32, WARP_NUM_PER_BLOCK, 1);
            quant_aligned_q_8_0<T, 32, BLOCK_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>>(static_cast<T *>(src->get_buffer()->ptr()),
                dst->get_buffer()->ptr(),M, N / BLOCK_SIZE);
            event->record(stream);
        }

    }

    template<typename T>
    void tff::kernel::QuantAligned<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), QuantAligned<T>::get_op_name().c_str());
        auto input_tensor = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        auto output_tensor = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        const auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(3, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream>>(4, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent>>(5, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent>>>(6, para_ptr);
        if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }
        if (input_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("input_tensor buffer is nullptr!");
            return;
        }
        if (output_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("output_tensor buffer is nullptr!");
            return;
        }

        const int M = input_tensor->get_shape()[1];
        const int N = input_tensor->get_shape()[0];
        quant_aligned<T>(M, N, input_tensor, output_tensor, stream, event, event_list);
    }

    template<typename T>
    std::string tff::kernel::QuantAligned<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(core::graph::TffOpType::TFF_OP_QUANTIZE_ALIGNED);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();

        return name;
    }

    template class tff::kernel::QuantAligned<Q8_0_ALIGNED>;
    REGISTER_OP_OBJECT(QuantAligned, Q8_0_ALIGNED);
}
