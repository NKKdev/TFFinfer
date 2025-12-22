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
    template<typename T>
    static T get_param_value(const int &index, std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr) {
        auto opt = para_ptr->get_param<T>(index);
        if (!opt.has_value()) {
            tff::log::Logger::error("Failed to get param[%d]", index);
            return T{};
        }
        return opt.value();
    }

    template<typename T>
    class XGemm : public base::OPCreatorBase<XGemm<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        //
        static std::string get_op_name();
    };

    template<typename T>
    class MemMap2Cpu : public base::OPCreatorBase<MemMap2Cpu<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class MemCpy : public base::OPCreatorBase<MemCpy<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class Embedding : public base::OPCreatorBase<Embedding<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class Mul : public base::OPCreatorBase<Mul<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    template<typename T>
    class Reshape : public base::OPCreatorBase<Reshape<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    template<typename T>
    class Rope : public base::OPCreatorBase<Rope<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    template<typename T>
    class FlashAttn : public base::OPCreatorBase<FlashAttn<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class Add : public base::OPCreatorBase<Add<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class RMSNorm : public base::OPCreatorBase<RMSNorm<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class MemRef : public base::OPCreatorBase<MemRef<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class DeQuantQ8 : public base::OPCreatorBase<DeQuantQ8<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    //
    template<typename T>
    class QuantQ8 : public base::OPCreatorBase<QuantQ8<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    template<typename T>
    class QuantQ8Reshape : public base::OPCreatorBase<QuantQ8Reshape<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    template<typename T>
    class QuantQ8MatMul : public base::OPCreatorBase<QuantQ8MatMul<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    template<typename T>
    class SetRow : public base::OPCreatorBase<SetRow<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };

    template<typename T>
    class GetRow : public base::OPCreatorBase<GetRow<T>, T> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        static std::string get_op_name();
    };
}
#endif //TFFINFER_XGEMM_H
