//
// Created by nkk on 2025/11/18.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T>
    class Rope<T,
                core::device::GPUTag> : public base::OPCreatorBase<Rope<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_ROPE;
        }
    };

    //
    template<typename T>
    void tff::kernel::Rope<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        //auto name = para_ptr->get_param<std::string>(0);
        //tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::Rope<T>::get_op_name().c_str());
    }

    template class tff::kernel::Rope<float, core::device::GPUTag>;
    template class tff::kernel::Rope<double, core::device::GPUTag>;
    template class tff::kernel::Rope<int32_t, core::device::GPUTag>;
    template class tff::kernel::Rope<int64_t, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(Rope, float, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Rope, double, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Rope, int32_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Rope, int64_t, core::device::GPUTag);
}
