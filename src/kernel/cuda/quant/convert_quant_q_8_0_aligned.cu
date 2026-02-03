//
// Created by nkk on 2026/1/28.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"

namespace tff::kernel {
    template<typename T, const int WARP_SIZE, const int BLOCK_SIZE>
    __global__ __forceinline__ void quant_aligned_q_8_0(const Q8_0 *__restrict__ src,
                                                        T *dst, const int M,
                                                        const int dst_stride_cnt) {
        const int g_thread_id = threadIdx.y * blockDim.x + threadIdx.x;
        const int warp_id = g_thread_id / WARP_SIZE;
        const int lane_id = g_thread_id % WARP_SIZE;

        const int row = blockIdx.y * blockDim.y + warp_id;
        const int col = blockIdx.x * blockDim.x + lane_id;
        const int start_dst_row = row;
        const int start_dst_col = col / BLOCK_SIZE;

        const int index = start_dst_row * dst_stride_cnt + start_dst_col;
        if (start_dst_row < M && start_dst_col < dst_stride_cnt) {
            auto src_val = &src[index];
            if (lane_id == 0) {
                dst[index].d = __half2float(src_val->d);
            }
            dst[index].qs[lane_id] = src_val->qs[lane_id];
        }
    }
#ifdef _DEBUG
    static void varify(std::string &filename, std::shared_ptr<core::memory::Tensor> &tensor) {
        switch (tensor->get_data_type()) {
            case core::memory::DataType::TFF_DATA_TYPE_F32: {
                std::vector<float> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<float> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                    for (int nn = 0; nn < tensor->get_shape()[0]; nn++) {
                        float delta = weight_gpu_result[mm * tensor->get_shape()[0] + nn] - weight_cpu_result[
                                          mm * tensor->get_shape()[0] + nn];
                        if (fabs(delta) > 0.001f) {
                            tff::log::Logger::error("error: m: %d n: %d, delta: %lf", mm, nn, delta);
                            throw std::runtime_error("error");
                        }
                    }
                }
                break;
            }
            case core::memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED: {
                std::vector<Q8_0> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] / Q8_0::BLOCK_SIZE * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<Q8_0_ALIGNED> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                    for (int nn = 0; nn < tensor->get_shape()[0] / Q8_0_ALIGNED::BLOCK_SIZE; nn++) {
                        float delta = weight_gpu_result[mm * tensor->get_shape()[0] / Q8_0_ALIGNED::BLOCK_SIZE + nn].d -
                                      __half2float(weight_cpu_result[
                                          mm * tensor->get_shape()[0] / Q8_0_ALIGNED::BLOCK_SIZE + nn].d);
                        if (fabs(delta) > 0.001f) {
                            tff::log::Logger::error("error: m: %d n: %d, delta: %lf", mm, nn, delta);
                            throw std::runtime_error("error");
                        }
                    }
                }
                break;
            }
            case core::memory::DataType::TFF_DATA_TYPE_Q8_0: {
                std::vector<Q8_0> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] / Q8_0::BLOCK_SIZE * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<Q8_0> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                    for (int nn = 0; nn < tensor->get_shape()[0] / Q8_0::BLOCK_SIZE; nn++) {
                        float delta = __half2float(weight_gpu_result[mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d) -
                                      __half2float(weight_cpu_result[
                                          mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d);
                        if (fabs(delta) > 0.001f) {
                            tff::log::Logger::error("error: m: %d n: %d, delta: %lf", mm, nn, delta);
                            throw std::runtime_error("error");
                        }
                    }
                }
                break;
            }
            default:
                break;
        }

        tff::log::Logger::info("layer node op varify (%s) success!", filename.c_str());
    }
#endif
    template<typename T>
    void quant_aligned(const int M, const int N,
                       std::shared_ptr<tff::core::memory::Tensor> &src,
                       std::shared_ptr<tff::core::memory::Tensor> &dst,
                       std::shared_ptr<core::device::DeviceStream> &stream) {
        if constexpr (std::is_same_v<T, Q8_0_ALIGNED>) {
            constexpr int BLOCK_SIZE = tff::core::quant::Q_8_0::BLOCK_SIZE;
            constexpr int VEC_M_DIM = 8;
            constexpr int WARP_NUM_PER_BLOCK = 8;

            dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + WARP_NUM_PER_BLOCK - 1) / WARP_NUM_PER_BLOCK, 1);
            dim3 block(32, WARP_NUM_PER_BLOCK, 1);
            quant_aligned_q_8_0<T, 32, BLOCK_SIZE><<<grid, block, 0, static_cast<cudaStream_t>(stream->
                        get_native_stream())>>>
                    (static_cast<Q8_0 *>(src->get_buffer()->ptr()),
                     static_cast<T *>(dst->get_buffer()->ptr()), M, N / BLOCK_SIZE);
        }
    }

    template<typename T>
    void tff::kernel::QuantAligned<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            0, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            1, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
               para_ptr->get_param_count() - 1, para_ptr);

        if (input_tensor == nullptr || input_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("input_tensor buffer is nullptr!");
            return;
        }
        if (output_tensor == nullptr || output_tensor->get_buffer() == nullptr) {
            tff::log::Logger::error("output_tensor buffer is nullptr!");
            return;
        }

        const int M = input_tensor->get_shape()[1];
        const int N = input_tensor->get_shape()[0];
        quant_aligned<T>(M, N, input_tensor, output_tensor, stream);

#ifdef _DEBUG
        // cudaDeviceSynchronize();
        // cudaStreamSynchronize(static_cast<cudaStream_t>(stream->
        //             get_native_stream()));
        // std::string filename = "";
        // if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_Q) {
        //     filename = "Qcur-0_src_0.ggml";
        // } else if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_K) {
        //     filename = "Kcur-0_src_0.ggml";
        // } else if (input_tensor->get_tensor_type() == core::memory::ModelTensorType::LLM_TENSOR_ATTN_V) {
        //     filename = "Vcur-0_src_0.ggml";
        // }
        // varify(filename, input_tensor);
        // varify(filename, output_tensor);
#endif
    }


    template class tff::kernel::QuantAligned<Q8_0_ALIGNED>;
    REGISTER_OP_OBJECT(QuantAligned, Q8_0_ALIGNED);
}
