//
// Created by nkk on 2025/12/16.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"


namespace tff::kernel {
    template<const int WARP_SIZE, const int BLOCK_SIZE>
    __global__ void quant_q_8_0_2d_reshape(const float *__restrict__ src,
                                           half *scale_ptr,
                                           int8_t *quant_ptr,
                                           const int M, const int N, const int ld, const int dst_stride_cnt) {
        const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = g_thread_id / WARP_SIZE;
        const int lane_id = g_thread_id % WARP_SIZE;

        const int row = blockIdx.y * blockDim.y + warp_id;
        const int col = blockIdx.x * blockDim.x + lane_id;
        const int start_dst_row = row;
        const int start_dst_col = col / BLOCK_SIZE;


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
                scale_ptr[index] = __float2half(d);
            }
            *(quant_ptr + start_dst_row * ld + col) = max_value == 0
                                                          ? 0
                                                          : static_cast<int8_t>(static_cast<int32_t>(roundf(x / d)));
        }
    }
    template<typename T>
    void quant_q_8_0_reshape(const int M, const int N,
        const T * src,
        void *dst) {

        if (std::is_same_v<T, float>) {
            constexpr int BLOCK_SIZE = tff::core::quant::Q_8_0::BLOCK_SIZE;
            constexpr int WARP_NUM_PER_BLOCK = 8;
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
            dim3 block(32, WARP_NUM_PER_BLOCK, 1);
            auto scaled_gpu = static_cast<half*>(dst);
            auto quant_gpu = static_cast<int8_t*>(dst + M * N / BLOCK_SIZE);
            quant_q_8_0_2d_reshape<32, BLOCK_SIZE><<<grid, block>>>(src, scaled_gpu,quant_gpu, M, N, N, N / BLOCK_SIZE);
        }else if (std::is_same_v<T, half>) {
            //todo half impl;
        }
    }
    template<typename T>
    void tff::kernel::QuantQ8Reshape<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), QuantQ8Reshape<T>::get_op_name().c_str());
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            1, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            2, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(3, para_ptr);

        if (input_tensors.size() != 1) {
            tff::log::Logger::error("QuantQ8Reshape kernel param is invalid!");
            return;
        }
        if (output_tensors.size() != 1) {
            tff::log::Logger::error("QuantQ8Reshape kernel param is invalid!");
            return;
        }

        auto input_tensor = input_tensors.at(0);
        if (input_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("input_tensor buffer is nullptr!");
            return;
        }
        auto output_tensor = output_tensors.at(0);
        if (output_tensor->get_buffer() == nullptr) {
            auto mem_buffer_pair = mem_buffer_manager_ptr->get_gpu_memory();
            if (mem_buffer_pair.second == nullptr) {
                tff::log::Logger::error("mem_buffer_pair buffer is nullptr!");
                return;
            }
            output_tensor->set_buffer_data(mem_buffer_pair.second, output_tensor->get_bytes(), mem_buffer_pair.first);
        }
        const int M = input_tensor->get_shape()[1];
        const int N = input_tensor->get_shape()[0];
        quant_q_8_0_reshape<T>(M, N, static_cast<T *>(input_tensor->get_buffer()->ptr()), output_tensor->get_buffer()->ptr());
    }

    template<typename T>
    std::string tff::kernel::QuantQ8Reshape<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_RESHAPE);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();

        return name;
    }

    template class tff::kernel::QuantQ8Reshape<float>;
    //template class tff::kernel::QuantQ8Reshape<half>;
    REGISTER_OP_OBJECT(QuantQ8Reshape, float);

    //REGISTER_OP_OBJECT(QuantQ8Reshape, half);
}
