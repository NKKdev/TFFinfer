//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_XGEMM_H
#define TFFINFER_XGEMM_H
#include "Buffer.h"
#include "core/global/ModelGlobalVar.h"
#include "TFFOPCreatorBase.h"
namespace tff::kernel {
    template <typename T>
    class XGemm :public base::OPCreatorBase<XGemm<T>, T>{
    public:
        static void compute(base::Layout layout,
                            base::Transpose a_transpose,  base::Transpose b_transpose,
                            size_t m,  size_t n,  size_t k,
                            float alpha,
                            Buffer<T> &a_buffer,  size_t a_offset,  size_t a_ld,
                            Buffer<T> &b_buffer,  size_t b_offset,  size_t b_ld,
                            float beta,
                            Buffer<T> &c_buffer,  size_t c_offset,  size_t c_ld);
        //
        static const char *get_op_name() {
            return core::global::TFF_OP_TYPE_MAP.find(tff::core::graph::TffOpType::TFF_OP_MUL_MAT)->second;
        }

    };


}
#endif //TFFINFER_XGEMM_H