//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_XGEMM_H
#define TFFINFER_XGEMM_H
#include "Buffer.h"
#include "core/global/ModelGlobalVar.h"
#include "TFFOPCreatorBase.h"
#include "global/ParamBaseObject.h"
#include "log/Logger.h"
namespace tff::kernel {
    template <typename T>
    class XGemm :public base::OPCreatorBase<XGemm<T>, T>{
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);
        //
        static std::string get_op_name() {
            auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MUL_MAT);
            if (it == core::global::TFF_OP_TYPE_MAP.end()) {
                tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
                return "";
            }
            return std::string(it->second) + DEVICE_BACKEND_TYPE_CPU;
        }

    };
    template <typename T>
    class MemMap2Cpu :public base::OPCreatorBase<MemMap2Cpu<T>, T> {
        public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);
        static std::string get_op_name() {
            auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MAP2CPU);
            if (it == core::global::TFF_OP_TYPE_MAP.end()) {
                tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
                return "";
            }
            return std::string(it->second) + DEVICE_BACKEND_TYPE_CPU;
        }
    };
    //
    template <typename T>
    class MemCpy :public base::OPCreatorBase<MemCpy<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);
        static std::string get_op_name() {
            auto it = core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MEM_CPY);
            if (it == core::global::TFF_OP_TYPE_MAP.end()) {
                tff::log::Logger::error("Op type not found in TFF_OP_TYPE_MAP");
                return "";
            }
            return std::string(it->second) + DEVICE_BACKEND_TYPE_CUDA;
        }
    };


}
#endif //TFFINFER_XGEMM_H