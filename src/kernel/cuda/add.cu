//
// Created by nkk on 2025/11/18.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"

namespace tff::kernel {
    template<typename T>
    __global__ __forceinline__ void add_kernel(const float *__restrict__) {

    }
    template<typename T>
    class Add<T,
                core::device::GPUTag> : public base::OPCreatorBase<Add<T, core::device::GPUTag>, T,
                core::device::GPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_ADD;
        }
    };

    //
    template<typename T>
    void tff::kernel::Add<T, core::device::GPUTag>::compute(
        std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {

    }

    template class tff::kernel::Add<float, core::device::GPUTag>;
    template class tff::kernel::Add<double, core::device::GPUTag>;
    template class tff::kernel::Add<int32_t, core::device::GPUTag>;
    template class tff::kernel::Add<int64_t, core::device::GPUTag>;
    template class tff::kernel::Add<Q8_0, core::device::GPUTag>;
    REGISTER_OP_OBJECT_DEVICE(Add, float, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Add, double, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Add, int32_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Add, int64_t, core::device::GPUTag);

    REGISTER_OP_OBJECT_DEVICE(Add, Q8_0, core::device::GPUTag);
}
