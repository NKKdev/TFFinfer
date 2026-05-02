//
// Created by nkk on 2026/2/13.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
#include "kernel/include/kernel_util.h"
#include "core/runtime/KVCache.h"

namespace tff::kernel {
    template<typename T>
    class GatherOP<T,
                core::device::GPUTag> : public base::OPCreatorBase<GatherOP<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_GATHER;
        }
    };

    template<typename T>
    void gather(std::vector<int> &indexs,
                std::shared_ptr<core::memory::Tensor> &input_tensor,
                std::shared_ptr<core::memory::Tensor> &output_tensor,
                std::shared_ptr<core::device::DeviceStream> &stream) {
        auto allocator = input_tensor->get_allocator();
        int i = 0;
        for (auto &index: indexs) {
            T *row_data_ptr = reinterpret_cast<T *>(reinterpret_cast<char *>(input_tensor->get_buffer()->ptr()) + index *
                                               input_tensor->get_strides()[1]);
            T *output_data_ptr = reinterpret_cast<T *>(
                reinterpret_cast<char *>(output_tensor->get_buffer()->ptr()) + i * output_tensor->get_strides()[1]);
            allocator->memcpy_async(row_data_ptr, output_data_ptr, input_tensor->get_strides()[1],
                                    core::memory::MemCpyKind::TFF_MEM_CPY_TYPE_DEVICE2DEVICE,
                                    stream->get_native_stream());
        }
    }

    template<typename T>
    void tff::kernel::GatherOP<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto row_indexs = kernel::base::get_param_value<std::vector<int> >(GatherOPBuilder::Params::RowIndex, para_ptr);
        auto input_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(
            GatherOPBuilder::Params::In, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<core::memory::Tensor> >(
            GatherOPBuilder::Params::Out, para_ptr);

        auto stream = kernel::base::get_param_value<std::shared_ptr<core::device::DeviceStream> >(
            kernel::builder::OpParamBuilderBase<MemOptOPBuilder>::CommonParams::Stream, para_ptr);
        if (output_tensor == nullptr || output_tensor->get_buffer() == nullptr) {
            return;
        }
        //gather<T>(row_indexs, input_tensor, output_tensor, stream);
    }


    template class tff::kernel::GatherOP<float, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(GatherOP, float, core::device::GPUTag);
}
