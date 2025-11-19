//
// Created by nkk on 2025/11/19.
//
#include "include/TFFOPCreator.h"
#include "model/base/ModelLoaderBase.h"
#include "model/FileLoader.h"
#include "runtime/LLMWeightMemManager.h"

namespace tff::kernel {
    template<typename T>
    void tff::kernel::MemRef<T>::compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        const auto &name = get_param_value<std::string>(0, para_ptr);
        tff::log::Logger::info("layer node %s op:%s compute!", name.c_str(), MemRef<T>::get_op_name().c_str());

        return;
    }

    template class tff::kernel::MemRef<float>;
    template class tff::kernel::MemRef<double>;
    template class tff::kernel::MemRef<int32_t>;
    REGISTER_OP_OBJECT(MemRef, float);
    REGISTER_OP_OBJECT(MemRef, double);
    REGISTER_OP_OBJECT(MemRef, int32_t);
}
