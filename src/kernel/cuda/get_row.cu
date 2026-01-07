//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
namespace tff::kernel {

    template<typename T1, typename T2>
    __global__ void get_row_f16(const int src_dim0, const int src_dim1,
        const T1 *__restrict__ src,
        const int64_t *idx,
        const int dst_dim0, const int dst_dim1,
        T2 *__restrict__ dst) {
        if (src_dim0 != dst_dim0) {
            return;
        }
        const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
        const int thread_x = thread_id % 32;
        const int warp_id = thread_id / 32;


    }
    template<typename T>
    void tff::kernel::GetRow<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), GetRow<T>::get_op_name().c_str());
        auto input_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            1, para_ptr);
        auto output_tensors = get_param_value<std::vector<std::shared_ptr<tff::core::memory::Tensor> > >(
            2, para_ptr);
        auto mem_buffer_manager_ptr = get_param_value<
            std::shared_ptr<
                tff::core::runtime::LLMMemManager> >(3, para_ptr);




    }
    template<typename T>
    std::string tff::kernel::GetRow<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_GET_ROWS);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;

        return name;
    }

    template class tff::kernel::GetRow<float>;
    template class tff::kernel::GetRow<half>;
    REGISTER_OP_OBJECT(GetRow, float);
    REGISTER_OP_OBJECT(GetRow, half);
}
