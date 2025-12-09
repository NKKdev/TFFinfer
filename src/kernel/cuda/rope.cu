//
// Created by nkk on 2025/11/18.
//

#include "device/cuda/cudaInc.h"
#include "kernel/include/TFFOPCreator.h"
namespace tff::kernel {
    //
    template<typename T>
    void tff::kernel::Rope<T>::compute( std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        //auto name = para_ptr->get_param<std::string>(0);
        //tff::log::Logger::info("layer node : %s, op:%s compute!",name.c_str(), tff::kernel::Rope<T>::get_op_name().c_str());
    }
    template<typename T>
    std::string tff::kernel::Rope<T>::get_op_name() {
        auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_ROPE);
        if (it == core::global::TFF_OP_TYPE_MAP.end()) {
            tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
            return "";
        }
        std::string name = std::string(it->second);
        name += std::string("_") + DEVICE_BACKEND_TYPE_CUDA;
        // if (std::is_same_v<T, float>) name += "_f32";
        // else if (std::is_same_v<T, double>) name += "_f64";
        // else if (std::is_same_v<T, int8_t>) name += "_i8";
        // else if (std::is_same_v<T, int16_t>) name += "_i16";
        return name;
    }
    template class tff::kernel::Rope<float>;
    template class tff::kernel::Rope<double>;
    template class tff::kernel::Rope<int32_t>;
    template class tff::kernel::Rope<int64_t>;
    REGISTER_OP_OBJECT(Rope, float);
    REGISTER_OP_OBJECT(Rope, double);
    REGISTER_OP_OBJECT(Rope, int32_t);
    REGISTER_OP_OBJECT(Rope, int64_t);
}