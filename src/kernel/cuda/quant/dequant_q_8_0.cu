//
// Created by nkk on 2025/12/17.
//
#include <fmt/core.h>

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T, const int BLOCK_SIZE>
    __global__ void dequant_q_8_0(const int M, const int N, const int ld,
                                  const tff::core::quant::Q_8_0 *src, T *dst,
                                  const int dst_stride_cnt) {
        const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = g_thread_id / 32;
        const int lane_id = g_thread_id % 32;

        const int row = blockIdx.y * blockDim.y + warp_id;
        const int col = blockIdx.x * blockDim.x + lane_id;
        const int start_dst_row = row;
        const int start_dst_col = col / BLOCK_SIZE;


        const int index = start_dst_row * dst_stride_cnt + start_dst_col;
        if (start_dst_row >= M && start_dst_col >= dst_stride_cnt) {
            return;
        }
        float scale = 0.0f;
        if (lane_id == 0) {
            scale = __half2float(src[index].d);
        }
        scale = __shfl_sync(0xFFFFFFFF, scale, 0);

        if (col < N && row < M) {
            if (std::is_same_v<T, half>) {
                dst[row * ld + col] = __float2half(static_cast<float>(src[index].qs[lane_id]) * scale);
            }else if (std::is_same_v<T, float>) {
                dst[row * ld + col] = static_cast<float>(src[index].qs[lane_id]) * scale;
            }

        }
    }

    template<typename T>
    void tff::kernel::DeQuantQ8<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {

    }

    template<typename T>
    std::string tff::kernel::DeQuantQ8<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_DEQUANTIZE_Q8);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();

        return name;
    }

    template class tff::kernel::DeQuantQ8<float>;
    template class tff::kernel::DeQuantQ8<half>;
    REGISTER_OP_OBJECT(DeQuantQ8, float);

    REGISTER_OP_OBJECT(DeQuantQ8, half);
}
