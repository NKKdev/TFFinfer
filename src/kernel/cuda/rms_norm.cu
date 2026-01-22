//
// Created by nkk on 2025/11/18.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "utils/util.h"

namespace tff::kernel {
    template<typename T, int block_size, int warp_size>
    __global__ void rms_norm_kernel_cuda_impl(
        const uint32_t magic,
        const uint32_t shift,
        const float eps,
        const int B,
        const int S,
        const int D,
        const T *const src,
        const T *const weight,
        T *const dst) {
        const int batch_index = blockIdx.y;
        const int row_index = blockIdx.x;
        const int thread_id = threadIdx.x;
        if (row_index > B * S) {
            return;
        }
        const int global_index = batch_index * S * D + row_index * D;
        const T *value_row_data = src + global_index;
        const T *weight_row_data = weight + global_index;
        T *value_dst = dst + global_index;
        //
        float sub_sum_seq = 0.0f;
#pragma unroll
        for (int d = thread_id; d < D / warp_size; d += block_size) {
            float value = *(value_row_data + d);
            sub_sum_seq += value * value;
        }
        //
#pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            sub_sum_seq += __shfl_xor_sync(0xffffffff, sub_sum_seq, offset, warp_size);
        }
        //
        float rms = rsqrtf(sub_sum_seq * tff::kernel::div_u32(D, magic, shift) + eps);
#pragma unroll
        for (int d = thread_id; d < D / warp_size; d += block_size) {
            *(value_dst + d) = *(value_row_data + d) * rms * *(weight_row_data + d);
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
                             std::vector<std::shared_ptr<core::device::DeviceEvent>> &wait_event_list) {
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
        auto div_magic = tff::utils::gen_magic_u32(src_dim0);
        const dim3 grid(src_dim1, src_dim2, 1);
        const dim3 block(32, 1, 1);
        for (auto &wait_event : wait_event_list) {
            stream->wait_event(wait_event->get_native_event());
        }
        rms_norm_kernel_cuda_impl<T, 32, 32><<<grid, block, 0, static_cast<cudaStream_t>(stream->get_native_stream())>>>(
            std::get<0>(div_magic),
            std::get<1>(div_magic),
            eps,
            src_dim2, src_dim1, src_dim0,
            (T *) input_tensor->get_buffer()->ptr(),
            (T *) weight_tensor->get_buffer()->ptr(),
            (T *) output_tensor->get_buffer()->ptr());
        event->record(stream);
    }

    //
    template<typename T>
    void tff::kernel::RMSNorm<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), RMSNorm<T>::get_op_name().c_str());
        auto eps = get_param_value<float>(1, para_ptr);
        auto weight = get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            2, para_ptr);
        auto x = get_param_value<std::shared_ptr<tff::core::memory::Tensor>>(
            3, para_ptr);
        auto output_tensors = get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            4, para_ptr);
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
