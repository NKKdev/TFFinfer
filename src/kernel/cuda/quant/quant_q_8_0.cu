//
// Created by nkk on 2025/12/16.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    template<const int WARP_SIZE, const int BLOCK_SIZE, const int WARP_NUM_PER_BLOCK>
    __global__ void quant_q_8_0(const float *__restrict__ src,
                                void *dst, const int64_t M, const int64_t N, const int ld, const int dst_stride_cnt) {
        const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = g_thread_id / WARP_SIZE;
        const int lane_id = g_thread_id % WARP_SIZE;

        const int start_block = blockIdx.x * WARP_NUM_PER_BLOCK * WARP_SIZE;
        const int g_index = start_block + warp_id * BLOCK_SIZE + lane_id;

        const int g_dst_index = blockIdx.x * WARP_NUM_PER_BLOCK + warp_id;
        auto *dst_ptr = static_cast<tff::core::quant::Q_8_0 *>(dst);

        float x = 0.0f;
        float max_value = 0.0f;
        if (g_index < N * M) {
            x = src[g_index];
            max_value = fabsf(x);
        }

#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
        }

        float d = max_value / 127.0f;
        if (g_dst_index < M * N / BLOCK_SIZE) {
            if (lane_id == 0) {
                dst_ptr[g_dst_index].d = __float2half(d);
            }
            dst_ptr[g_dst_index].qs[lane_id] = max_value == 0
                                                   ? 0
                                                   : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
        }
    }

    template<const int WARP_SIZE, const int BLOCK_SIZE>
    __global__ void quant_q_8_0_2d(const float *__restrict__ src,
                                   void *dst, const int M, const int N, const int ld, const int dst_stride_cnt) {
        const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = g_thread_id / WARP_SIZE;
        const int lane_id = g_thread_id % WARP_SIZE;

        const int row = blockIdx.y * blockDim.y + warp_id;
        const int col = blockIdx.x * blockDim.x + lane_id;
        const int start_dst_row = row;
        const int start_dst_col = col / BLOCK_SIZE;
        auto *dst_ptr = static_cast<tff::core::quant::Q_8_0 *>(dst);

        float x = 0.0f;
        float max_value = 0.0f;
        if (col < N && row < M) {
            x = src[row * ld + col];
            max_value = fabsf(x);
        }

#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_xor_sync(0xffffffff, max_value, offset, WARP_SIZE));
        }
        float d = max_value / 127.0f;
        const int index = start_dst_row * dst_stride_cnt + start_dst_col;
        if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
            if (lane_id == 0) {
                dst_ptr[index].d = __float2half(d);
            }
            dst_ptr[index].qs[lane_id] = max_value == 0 ? 0 : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
        }
    }
    template<typename T>
    void quant(const int M, const int N,
        std::shared_ptr<tff::core::memory::Tensor> &src,
        std::shared_ptr<tff::core::memory::Tensor> &dst) {
        if (std::is_same_v<T, Q8_0>) {
            constexpr int BLOCK_SIZE = tff::core::quant::Q_8_0::BLOCK_SIZE;
            constexpr int VEC_M_DIM = 8;
            constexpr int WARP_NUM_PER_BLOCK = 8;
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
            dim3 block(32, WARP_NUM_PER_BLOCK, 1);
            quant_q_8_0_2d<32, BLOCK_SIZE><<<grid, block>>>(static_cast<float *>(src->get_buffer()->ptr()),
                dst->get_buffer()->ptr(), M, N, N, N / BLOCK_SIZE);
        }else if (std::is_same_v<T, core::quant::Q_8_1>) {
            //todo Q_8_1 impl;
        }
    }
    template<typename T>
    void tff::kernel::Quant<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), Quant<T>::get_op_name().c_str());
        auto input_tensor = get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            1, para_ptr);
        auto output_tensor = get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            2, para_ptr);
        const auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(3, para_ptr);

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
        quant<T>(M, N, input_tensor, output_tensor);

    }

    template<typename T>
    std::string tff::kernel::Quant<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();

        return name;
    }

    template class tff::kernel::Quant<Q8_0>;
    REGISTER_OP_OBJECT(Quant, Q8_0);
}
