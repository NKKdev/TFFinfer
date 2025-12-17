//
// Created by nkk on 2025/12/17.
//
#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    template<typename T>
    void tff::kernel::SetRow<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {

    }

    template<typename T>
    std::string tff::kernel::SetRow<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_SET_ROWS);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA + tff::core::global::get_type_suffix<T>();;

        return name;
    }

    template class tff::kernel::SetRow<float>;
    template class tff::kernel::SetRow<half>;

    REGISTER_OP_OBJECT(SetRow, float);

    REGISTER_OP_OBJECT(SetRow, half);
}
