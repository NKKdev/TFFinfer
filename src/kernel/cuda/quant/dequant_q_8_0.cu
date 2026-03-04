//
// Created by nkk on 2025/12/17.
//

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
    class DeQuant<T, core::device::GPUTag> : public base::OPCreatorBase<DeQuant<T, core::device::GPUTag>, T, core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_DEQUANTIZE_Q8;
        }
    };
    template<typename T>
    void tff::kernel::DeQuant<T, core::device::GPUTag>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {

    }


    template class tff::kernel::DeQuant<float, core::device::GPUTag>;
    template class tff::kernel::DeQuant<half, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(DeQuant, float, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(DeQuant, half, core::device::GPUTag);
}
