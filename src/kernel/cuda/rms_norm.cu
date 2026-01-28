//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "utils/util.h"

namespace tff::kernel {
    template<typename T, int BLOCK_DIM_M, int WARP_SIZE>
    __global__ void rms_norm_kernel_cuda_impl(
        const float eps,
        const int B,
        const int S,
        const int D,
        const T *const src,
        const T *const weight,
        T *const dst) {
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
        const int thread_x = thread_id % WARP_SIZE;
        const int warp_id = thread_id / WARP_SIZE;
        const int batch_index = blockIdx.y;
        const int start_row = blockIdx.x * BLOCK_DIM_M;
        const int VEC_DIM_M = BLOCK_DIM_M / blockDim.y;


#pragma unroll
        for (int mm = 0;mm < VEC_DIM_M; mm++) {
            const int row_index = start_row + warp_id + mm * BLOCK_DIM_M / VEC_DIM_M;
            if (row_index >= S) {
                continue;
            }
            float sub_sum_seq = 0.0f;
            float reg[128] = {0.0f};
#pragma unroll
            for (int d = thread_x; d < D; d += WARP_SIZE) {
                reg[d / WARP_SIZE] = src[batch_index * S * D + row_index * D + d];
                sub_sum_seq += reg[d / WARP_SIZE] * reg[d / WARP_SIZE];
            }
            //
#pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                sub_sum_seq += __shfl_xor_sync(0xffffffff, sub_sum_seq, offset, WARP_SIZE);
            }

            float mean = sub_sum_seq / D;
            //
            float rms = rsqrtf(mean + eps);
#pragma unroll
            for (int d = thread_x; d < D; d += WARP_SIZE) {
                dst[batch_index * S * D + row_index * D + d] = reg[d / WARP_SIZE] * rms * weight[d];
            }

        }
    }

    template<typename T>
    static void rms_norm_kernel_cuda(const float &eps,
                                     std::shared_ptr<tff::core::memory::Tensor> &weight,
                                     std::shared_ptr<tff::core::memory::Tensor> &x,
                                     std::shared_ptr<tff::core::memory::Tensor> &dst,
                                     std::shared_ptr<core::runtime::LLMMemManager> &mem_buffer_manager_ptr,
                                     std::shared_ptr<core::device::DeviceStream> &stream,
                                     std::shared_ptr<core::device::DeviceEvent> &event,
                                     std::vector<std::shared_ptr<core::device::DeviceEvent> > &wait_event_list) {
        auto &input_tensor = x;
        auto &weight_tensor = weight;
        auto &output_tensor = dst;
        auto &src_shape = input_tensor->get_shape();
        const int src_dim0 = src_shape[0]; //D
        const int src_dim1 = src_shape[1]; //S
        const int src_dim2 = 1; //B

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
        const dim3 grid((src_dim1 + BLOCK_DIM_M - 1) / BLOCK_DIM_M, src_dim2, 1);
        const dim3 block(32, 8, 1);
        for (auto &wait_event: wait_event_list) {
            stream->wait_event(wait_event->get_native_event());
        }
        rms_norm_kernel_cuda_impl<T, 32, 32><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>
                >(eps,
                    src_dim2, src_dim1, src_dim0,
                    (T *) input_tensor->get_buffer()->ptr(),
                    (T *) weight_tensor->get_buffer()->ptr(),
                    (T *) output_tensor->get_buffer()->ptr());
        event->record(stream);

#ifdef _DEBUG
        // std::vector<float> cpu_src;
        // cpu_src.resize(
        //     input_tensor->get_shape()[0] * input_tensor->get_shape()[1] * input_tensor->get_shape()[2] *
        //     input_tensor->get_shape()[3]);
        // load_tensor_raw("norm-0_src_0.ggml", cpu_src.data());
        // std::vector<float> gpu_src;
        // gpu_src.resize(
        //     input_tensor->get_shape()[0] * input_tensor->get_shape()[1] * input_tensor->get_shape()[2] *
        //     input_tensor->get_shape()[3]);
        // input_tensor->get_allocator()->memcopy(input_tensor->get_buffer()->ptr(), gpu_src.data(),
        //     input_tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);
        //
        // for (int mm = 0; mm < input_tensor->get_shape()[1]; mm++) {
        //     for (int nn = 0; nn < input_tensor->get_shape()[0]; nn++) {
        //         float delta = gpu_src[mm * input_tensor->get_shape()[0] + nn] - cpu_src[mm * input_tensor->get_shape()[0] + nn];
        //         if (delta > 0.001f) {
        //             tff::log::Logger::error("error: m: %d n: %d, delta: %lf", mm, nn, delta);
        //             throw std::runtime_error("error");
        //         }
        //     }
        // }
        // tff::log::Logger::info("layer node op input success!");
        //
        // std::vector<float> cpu_result;
        // cpu_result.resize(
        //     output_tensor->get_shape()[0] * output_tensor->get_shape()[1] * output_tensor->get_shape()[2] *
        //     output_tensor->get_shape()[3]);
        // load_tensor_raw("attn_norm-0_result.ggml", cpu_result.data());
        // std::vector<float> gpu_result;
        // gpu_result.resize(
        //     output_tensor->get_shape()[0] * output_tensor->get_shape()[1] * output_tensor->get_shape()[2] *
        //     output_tensor->get_shape()[3]);
        // output_tensor->get_allocator()->memcopy(output_tensor->get_buffer()->ptr(), gpu_result.data(),
        //     output_tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);
        //
        // for (int mm = 0; mm < output_tensor->get_shape()[1]; mm++) {
        //     for (int nn = 0; nn < output_tensor->get_shape()[0]; nn++) {
        //         float delta = gpu_result[mm * output_tensor->get_shape()[0] + nn] - cpu_result[mm * output_tensor->get_shape()[0] + nn];
        //         if (fabs(delta) > 0.001f) {
        //             tff::log::Logger::error("error: m: %d n: %d, delta: %lf", mm, nn, delta);
        //             throw std::runtime_error("error");
        //         }
        //     }
        // }
        // tff::log::Logger::info("layer node op compute success!");
#endif
    }

    //
    template<typename T>
    void tff::kernel::RMSNorm<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), RMSNorm<T>::get_op_name().c_str());
        auto eps = get_param_value<float>(1, para_ptr);
        auto weight = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            2, para_ptr);
        auto x = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            3, para_ptr);
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            4, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(5, para_ptr);
        auto stream = get_param_value<std::shared_ptr<core::device::DeviceStream> >(6, para_ptr);
        auto event = get_param_value<std::shared_ptr<core::device::DeviceEvent> >(7, para_ptr);
        auto event_list = get_param_value<std::vector<std::shared_ptr<core::device::DeviceEvent> > >(8, para_ptr);
        if (stream == nullptr || event == nullptr || mem_buffer_manager_ptr == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }

        if (weight == nullptr || x == nullptr || output_tensors == nullptr) {
            tff::log::Logger::error("kernel (%s) param is invalid!", name.c_str());
            return;
        }

        //
        rms_norm_kernel_cuda<T>(eps, weight, x, output_tensors, mem_buffer_manager_ptr, stream, event, event_list);
    }

    template<typename T>
    std::string tff::kernel::RMSNorm<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_RMS_NORM);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;
        return name;
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
