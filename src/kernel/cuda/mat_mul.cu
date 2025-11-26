//
// Created by nkk on 2025/11/3.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T, int block_size, int warp_size>
    __global__ void gemm_kernel_cuda_impl(
        const int N,
        const int M,
        const int K,
        const T *const a,
        const T *const b,
        T *const c) {

    }
    template<typename T>
    static void gemm_kernel_cuda(std::set<std::shared_ptr<tff::core::memory::Tensor>,
                                     tff::core::memory::Tensor::TensorCompare> &src,
                                 std::set<std::shared_ptr<tff::core::memory::Tensor>,
                                     tff::core::memory::Tensor::TensorCompare> &dst,
                                 std::shared_ptr<core::runtime::LLMWeightMemManager> &mem_buffer_manager_ptr) {
        auto &input_tensor_a = *src.begin();
        auto &input_tensor_b = *src.rbegin();
        auto &output_tensor = *dst.begin();
        auto &a_shape = input_tensor_a->get_shape();
        int D = 0;
        int S = 0;
        int B = 0;
        if (a_shape.size() == 2) {
            //单个batch
            D = a_shape[0]; //D
            S = a_shape[1]; //S
            B = 1; //B
        } else if (a_shape.size() == 3) {
            //多个batch
            D = a_shape[0]; //D
            S = a_shape[1]; //S
            B = a_shape[2]; //B
        }


        auto mem_buffer = mem_buffer_manager_ptr->get_gpu_memory();
        if (mem_buffer.second == nullptr) {
            tff::log::Logger::error("rms_norm_kernel_cuda: mem_buffer_manager_ptr is nullptr!");
            return;
        }
        output_tensor->set_buffer_data(mem_buffer.second, output_tensor->get_bytes(), mem_buffer.first);
    }

    template<typename T>
    void tff::kernel::XGemm<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), XGemm<T>::get_op_name().c_str());
        auto input_tensors = get_param_value<std::set<std::shared_ptr<tff::core::memory::Tensor>,
            tff::core::memory::Tensor::TensorCompare> >(
            1, para_ptr);
        auto output_tensors = get_param_value<std::set<std::shared_ptr<tff::core::memory::Tensor>,
            tff::core::memory::Tensor::TensorCompare> >(
            2, para_ptr);
        std::shared_ptr<core::runtime::LLMWeightMemManager> mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMWeightMemManager> >(3, para_ptr);

        if (input_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
        if (output_tensors.size() != 1) {
            tff::log::Logger::error("memcpy kernel param is invalid!");
            return;
        }
    }

    template class tff::kernel::XGemm<float>;
    template class tff::kernel::XGemm<double>;
    REGISTER_OP_OBJECT(XGemm, float);

    REGISTER_OP_OBJECT(XGemm, double);
}
