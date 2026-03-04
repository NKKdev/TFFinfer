//
// Created by nkk on 2025/11/14.
//

#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/MemManager.h"

namespace tff::kernel {
    template<typename T>
    static void mem_map2cpu_kernel_cpu(const size_t &model_file_index,
                                       const size_t &offset,
                                       const double &data_size,
                                       const std::shared_ptr<tff::core::model::ModelLoaderBase> &model_loader_ptr,
                                       std::shared_ptr<tff::core::memory::Tensor> &outputs) {
        if (model_file_index < 0) {
            tff::log::Logger::error("Model file index is invalid");
            return;
        }
        if (data_size < 0) {
            tff::log::Logger::error("Data size is invalid");
            return;
        }
        if (offset < 0) {
            tff::log::Logger::error("Offset is invalid");
            return;
        }

        auto file_map_ptr = model_loader_ptr->get_file_map(model_file_index);
        if (file_map_ptr == nullptr) {
            tff::log::Logger::error("Failed to get file map");
            return;
        }
        auto buffer = (uint8_t *) file_map_ptr->addr() + offset;
        outputs->set_buffer_data(buffer, data_size);
    }

    template<typename T>
    class Map2Cpu<T,
                core::device::CPUTag> : public base::OPCreatorBase<Map2Cpu<T, core::device::CPUTag>, T,
                core::device::CPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MAP2CPU;
        }
    };

    template<typename T>
    void tff::kernel::Map2Cpu<T, core::device::CPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto model_file_index = kernel::base::get_param_value<size_t>(
            Map2CpuBuilder::Params::FileIdx, para_ptr);
        const auto offset = kernel::base::get_param_value<size_t>(Map2CpuBuilder::Params::Offset, para_ptr);
        const auto data_size = kernel::base::get_param_value<double>(Map2CpuBuilder::Params::Size, para_ptr);
        const auto model_loader_ptr = kernel::base::get_param_value<std::shared_ptr<
            tff::core::model::ModelLoaderBase> >(Map2CpuBuilder::Params::ModelCtx, para_ptr);
        auto output_tensor = kernel::base::get_param_value<std::shared_ptr<tff::core::memory::Tensor> >(
            Map2CpuBuilder::Params::Out, para_ptr);

        mem_map2cpu_kernel_cpu<T>(model_file_index, offset, data_size, model_loader_ptr, output_tensor);
    }

    template class tff::kernel::Map2Cpu<float, core::device::CPUTag>;
    template class tff::kernel::Map2Cpu<double, core::device::CPUTag>;
    template class tff::kernel::Map2Cpu<int32_t, core::device::CPUTag>;
    template class tff::kernel::Map2Cpu<Q8_0, core::device::CPUTag>;
    REGISTER_OP_OBJECT_DEVICE(Map2Cpu, float, core::device::CPUTag);

    REGISTER_OP_OBJECT_DEVICE(Map2Cpu, double, core::device::CPUTag);

    REGISTER_OP_OBJECT_DEVICE(Map2Cpu, int32_t, core::device::CPUTag);

    REGISTER_OP_OBJECT_DEVICE(Map2Cpu, Q8_0, core::device::CPUTag);
}
