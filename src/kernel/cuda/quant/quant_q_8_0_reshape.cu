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
    void tff::kernel::QuantQ8Reshape<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
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
    template class tff::kernel::QuantQ8Reshape<half>;
    REGISTER_OP_OBJECT(QuantQ8Reshape, float);

    REGISTER_OP_OBJECT(QuantQ8Reshape, half);
}
