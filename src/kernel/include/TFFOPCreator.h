//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_XGEMM_H
#define TFFINFER_XGEMM_H
#include "core/global/ModelGlobalVar.h"
#include "TFFOPCreatorBase.h"
#include "global/ParamBaseObject.h"

namespace tff::kernel {
    template<typename T>
    class XGemm : public base::OPCreatorBase<XGemm<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        //
        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MUL_MAT;
        }
    };

    template<typename T>
    class MemMap2Cpu : public base::OPCreatorBase<MemMap2Cpu<T>, T, core::device::CPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MAP2CPU;
        }
    };

    //
    template<typename T>
    class MemCpy : public base::OPCreatorBase<MemCpy<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MEM_CPY;
        }
    };

    //
    template<typename T>
    class Embedding : public base::OPCreatorBase<Embedding<T>, T, core::device::CPUTag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_EMBEDDING;
        }
    };

    //
    template<typename T>
    class Mul : public base::OPCreatorBase<Mul<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MUL;
        }
    };

    template<typename T>
    class Reshape : public base::OPCreatorBase<Reshape<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_RESHAPE;
        }
    };

    template<typename T>
    class Rope : public base::OPCreatorBase<Rope<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_ROPE;
        }
    };

    template<typename T>
    class FlashAttn : public base::OPCreatorBase<FlashAttn<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_FLASH_ATTN_EXT;
        }
    };

    //
    template<typename T>
    class PagedFlashAttn : public base::OPCreatorBase<PagedFlashAttn<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_FLASH_ATTN_PAGED;
        }
    };

    //
    template<typename T>
    class Add : public base::OPCreatorBase<Add<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_ADD;
        }
    };

    //
    template<typename T>
    class RMSNorm : public base::OPCreatorBase<RMSNorm<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_RMS_NORM;
        }
    };

    //
    template<typename T>
    class MemRef : public base::OPCreatorBase<MemRef<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_MEM_REF;
        }
    };

    //
    template<typename T>
    class DeQuantQ8 : public base::OPCreatorBase<DeQuantQ8<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_DEQUANTIZE_Q8;
        }
    };

    //
    template<typename T>
    class Quant : public base::OPCreatorBase<Quant<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_QUANTIZE_Q8;
        }
    };
    template<typename T>
    class QuantAligned : public base::OPCreatorBase<QuantAligned<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_QUANTIZE_ALIGNED;
        }
    };

    template<typename T>
    class QuantQ8Reshape : public base::OPCreatorBase<QuantQ8Reshape<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_RESHAPE;
        }
    };

    template<typename T>
    class QuantQ8MatMulReshape : public base::OPCreatorBase<QuantQ8MatMulReshape<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_MATMUL_RESHAPE;
        }
    };

    template<typename T>
    class QuantQ8MatMul : public base::OPCreatorBase<QuantQ8MatMul<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_QUANTIZE_Q8_MATMUL;
        }
    };

    template<typename T>
    class SetRow : public base::OPCreatorBase<SetRow<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_SET_ROWS;
        }
    };

    template<typename T>
    class GetRow : public base::OPCreatorBase<GetRow<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_GET_ROWS;
        }
    };

    template<typename T>
    class PreRopeTable : public base::OPCreatorBase<PreRopeTable<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_PRE_ROPE_TABLE;
        }
    };

    template<typename T>
    class UnaryOP : public base::OPCreatorBase<UnaryOP<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_UNARY;
        }
    };

    //
    template<typename T>
    class MaskOP : public base::OPCreatorBase<MaskOP<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_ATTN_MASK;
        }
    };
    //
    template<typename T>
    class ViewOP : public base::OPCreatorBase<ViewOP<T>, T, core::device::CUDATag> {
    public:
        static void compute(std::shared_ptr<tff::core::global::ParamBaseObject> &para_ptr);

        inline static core::graph::TffOpType op_type() {
            return core::graph::TffOpType::TFF_OP_VIEW;
        }
    };
}
#endif //TFFINFER_XGEMM_H
