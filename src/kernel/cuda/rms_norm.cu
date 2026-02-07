//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {
    template<typename T, int BLOCK_DIM_M, int WARP_SIZE>
    __global__ void rms_norm_kernel_cuda_impl(
        const float eps,
        const int dim0, const int dim1, const int dim2, const int dim3,
        const T *const src,
        const T *const weight,
        T *const dst) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        const int thread_x = thread_id % WARP_SIZE;
        const int warp_id = thread_id / WARP_SIZE;
        const int batch_index = blockIdx.z;
        const int head_index = blockIdx.x;
        const int start_row = blockIdx.y * BLOCK_DIM_M;
        const int VEC_DIM_M = BLOCK_DIM_M / blockDim.y;
        const T *src_ptr = src + batch_index * dim2 * dim1 * dim0 + start_row * dim1 * dim0 + head_index * dim0;
        T *dst_ptr = dst + batch_index * dim2 * dim1 * dim0 + start_row * dim1 * dim0 + head_index * dim0;
#pragma unroll
        for (int mm = 0;mm < VEC_DIM_M; mm++) {
            const int row_index = warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
            if ((start_row + row_index) >= dim2) {
                continue;
            }
            float sub_sum_seq = 0.0f;
#pragma unroll
            for (int d = thread_x; d < dim0; d += WARP_SIZE) {
                T reg = src_ptr[row_index * dim1 * dim0 + d];
                sub_sum_seq = fmaf(reg, reg, sub_sum_seq);
            }
            //
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                sub_sum_seq += __shfl_xor_sync(0xffffffff, sub_sum_seq, offset, WARP_SIZE);
            }

            float mean = sub_sum_seq / dim0;
            //
            float rms = rsqrtf(mean + eps);
#pragma unroll
            for (int d = thread_x; d < dim0; d += WARP_SIZE) {
                dst_ptr[row_index * dim1 * dim0 + d] = src_ptr[row_index * dim1 * dim0 + d] * rms * weight[d];
            }

        }
    }

    template<typename T>
    static void rms_norm_kernel_cuda(const float &eps,
                                     std::shared_ptr<tff::core::memory::Tensor> &weight,
                                     std::shared_ptr<tff::core::memory::Tensor> &x,
                                     std::shared_ptr<tff::core::memory::Tensor> &dst,
                                     std::shared_ptr<core::device::DeviceStream> &stream) {
        auto &input_tensor = x;
        auto &weight_tensor = weight;
        auto &output_tensor = dst;
        auto &src_shape = input_tensor->get_shape();
        const int d = src_shape[0]; //D
        const int h = src_shape[1]; //H
        const int s = src_shape[2]; //S
        const int b = src_shape[3]; //B

        if (input_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("rms_norm_kernel_cuda: input_tensor is nullptr!");
            return;
        }
        if (weight_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("rms_norm_kernel_cuda: weight_tensor is nullptr!");
            return;
        }
        if (output_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("rms_norm_kernel_cuda: output_tensor is nullptr!");
            return;
        }
        constexpr int BLOCK_DIM_M = 32;
        const dim3 grid(h, (s + BLOCK_DIM_M - 1) / BLOCK_DIM_M, b);
        const dim3 block(32, 8, 1);

        rms_norm_kernel_cuda_impl<T, 32, 32><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>
                >(eps,
                    d, h, s, b,
                    (T *) input_tensor->get_buffer()->ptr(),
                    (T *) weight_tensor->get_buffer()->ptr(),
                    (T *) output_tensor->get_buffer()->ptr());
    }

    //
    template<typename T>
    void tff::kernel::RMSNorm<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto eps = kernel::base::get_param_value<float>(0, para_ptr);
        auto weight = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);
        auto x = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            3, para_ptr);
        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
                        para_ptr->get_param_count() - 1, para_ptr);
        if (weight == nullptr || x == nullptr || output_tensor == nullptr) {
            return;
        }

        //
        rms_norm_kernel_cuda<T>(eps, weight, x, output_tensor, stream);
#ifdef _DEBUG1
        stream->synchronize();
        const auto &name = kernel::base::get_param_value<std::string>(para_ptr->get_param_count() - 5, para_ptr);
        std::string filename = "";
        if (name == "blk.0.attn_q_norm_rms_norm") {
            filename = "Qcur_normed-0_result.ggml";
            varify(filename, output_tensor);
        }else if (name == "blk.0.attn_k_norm_rms_norm") {
            filename = "Kcur_normed-0_result.ggml";
            varify(filename, output_tensor);
        }else if (name == "blk.0.attn_norm_rms_norm") {
            filename = "attn_norm-0_result.ggml";
            varify(filename, output_tensor);
        }

#endif
    }

    template class tff::kernel::RMSNorm<float>;
    template class tff::kernel::RMSNorm<double>;
    template class tff::kernel::RMSNorm<int32_t>;
    template class tff::kernel::RMSNorm<int64_t>;
    REGISTER_OP_OBJECT(RMSNorm, float);

    REGISTER_OP_OBJECT(RMSNorm, double);

    REGISTER_OP_OBJECT(RMSNorm, int32_t);

    REGISTER_OP_OBJECT(RMSNorm, int64_t);
}
