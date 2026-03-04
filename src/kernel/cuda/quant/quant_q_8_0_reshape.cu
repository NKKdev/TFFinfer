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
                             const float *src,
                             T *dst) {
        if (std::is_same_v<T, float>) {
            constexpr int BLOCK_SIZE = tff::core::quant::Q_8_0::BLOCK_SIZE;
            constexpr int WARP_NUM_PER_BLOCK = 8;
            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
            dim3 block(32, WARP_NUM_PER_BLOCK, 1);
            auto scaled_gpu = static_cast<half *>(static_cast<void*>(dst));
            auto quant_gpu = static_cast<int8_t *>(static_cast<void*>(dst) + M * N / BLOCK_SIZE);
            quant_q_8_0_2d_reshape<32, BLOCK_SIZE><<<grid, block>>
                    >(src, scaled_gpu, quant_gpu, M, N, N, N / BLOCK_SIZE);
        } else if (std::is_same_v<T, half>) {
            //todo half impl;
        }
    }

    template<typename T>
    class QuantReshape<T, core::device::GPUTag> : public base::OPCreatorBase<QuantReshape<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_RESHAPE;
        }
    };

    template<typename T>
    void tff::kernel::QuantReshape<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        // auto input_tensors = kernel::base::get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
        //     0, para_ptr);
        // auto output_tensors = kernel::base::get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
        //     1, para_ptr);
        //
        // if (input_tensors.size() != 1) {
        //     tff::log::Logger::error("QuantQ8Reshape kernel param is invalid!");
        //     return;
        // }
        // if (output_tensors.size() != 1) {
        //     tff::log::Logger::error("QuantQ8Reshape kernel param is invalid!");
        //     return;
        // }
        //
        // auto input_tensor = input_tensors.at(0);
        // if (input_tensor->get_buffer() == nullptr) {
        //     tff::log::Logger::error("input_tensor buffer is nullptr!");
        //     return;
        // }
        // auto output_tensor = output_tensors.at(0);
        // if (output_tensor->get_buffer() == nullptr) {
        //     //auto mem_buffer_pair = mem_buffer_manager_ptr->get_gpu_memory();
        //     // if (mem_buffer_pair.second == nullptr) {
        //     //     tff::log::Logger::error("mem_buffer_pair buffer is nullptr!");
        //     //     return;
        //     // }
        //     // output_tensor->set_buffer_data(mem_buffer_pair.second, output_tensor->get_bytes(), mem_buffer_pair.first);
        // }
        // const int M = input_tensor->get_shape()[1];
        // const int N = input_tensor->get_shape()[0];
        // quant_q_8_0_reshape<T>(M, N, static_cast<float *>(input_tensor->get_buffer()->ptr()),
        //                        static_cast<T *>(output_tensor->get_buffer()->ptr()));
    }

    template class tff::kernel::QuantReshape<Q8_0, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(QuantReshape, Q8_0, core::device::GPUTag);
}
