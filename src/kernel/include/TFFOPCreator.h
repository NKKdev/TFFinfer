//
// Created by nkk on 2025/11/3.
//

#ifndef TFFINFER_XGEMM_H
#define TFFINFER_XGEMM_H
#include "core/global/ModelGlobalVar.h"
#include "kernel/include/TFFOPCreatorBase.h"
#include "Builder.h"

namespace tff::kernel {
    /**
     * @brief Map2cpu算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Map2Cpu;
    /**
     * @brief Map2gpu算子参数构建器
     */
    class Map2CpuBuilder
            : public builder::OpParamBuilderBase<Map2CpuBuilder> {
    public:
        struct Params {
            static constexpr const char *ModelCtx = "model_ctx";
            static constexpr const char *FileIdx = "file_idx";
            static constexpr const char *Offset = "offset";
            static constexpr const char *Size = "size";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        Map2CpuBuilder &model_ctx(T &&value) {
            return set(Params::ModelCtx, value);
        }

        template<typename T>
        Map2CpuBuilder &file_idx(T &&value) {
            return set(Params::FileIdx, value);
        }

        template<typename T>
        Map2CpuBuilder &offset(T &&value) {
            return set(Params::Offset, value);
        }

        template<typename T>
        Map2CpuBuilder &size(T &&value) {
            return set(Params::Size, value);
        }

        template<typename T>
        Map2CpuBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        Map2CpuBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T model_ctx() {
            return get<T>(Params::ModelCtx);
        }

        template<typename T>
        T file_idx() {
            return get<T>(Params::FileIdx);
        }

        template<typename T>
        T offset() {
            return get<T>(Params::Offset);
        }

        template<typename T>
        T size() {
            return get<T>(Params::Size);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief MemCpy算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class MemCpy;
    /**
     * @brief MemCpy算子参数构建器
     */
    class MemCpyBuilder
            : public builder::OpParamBuilderBase<MemCpyBuilder> {
    public:
        struct Params {
            static constexpr const char *MemCpyKind = "memcpy_kind";
            static constexpr const char *SourceDeviceId = "source_id";
            static constexpr const char *DestDeviceId = "dest_id";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        MemCpyBuilder &memcpy_kind(T &&value) {
            return set(Params::MemCpyKind, value);
        }

        template<typename T>
        MemCpyBuilder &source_id(T &&value) {
            return set(Params::SourceDeviceId, value);
        }

        template<typename T>
        MemCpyBuilder &dest_id(T &&value) {
            return set(Params::DestDeviceId, value);
        }

        template<typename T>
        MemCpyBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        MemCpyBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T memcpy_kind() {
            return get<T>(Params::MemCpyKind);
        }

        template<typename T>
        T source_id() {
            return get<T>(Params::SourceDeviceId);
        }

        template<typename T>
        T dest_id() {
            return get<T>(Params::DestDeviceId);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }

        template<typename T>
        T out() {
            return get<T>(Params::Out);
        }
    };

    /**
     * @brief Embedding算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Embedding;
    /**
     * @brief Embedding算子参数构建器
     */
    class EmbeddingBuilder
            : public builder::OpParamBuilderBase<EmbeddingBuilder> {
    public:
        struct Params {
            static constexpr const char *InputToken = "input_token";
            static constexpr const char *Weight = "weight";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        EmbeddingBuilder &input_token(T &&value) {
            return set(Params::InputToken, value);
        }

        template<typename T>
        EmbeddingBuilder &weight(T &&value) {
            return set(Params::Weight, value);
        }

        template<typename T>
        EmbeddingBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T input_token() {
            return get<T>(Params::InputToken);
        }

        template<typename T>
        T weight() {
            return get<T>(Params::Weight);
        }
    };

    /**
     * @brief Mul算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Mul;
    /**
     * @brief Mul算子参数构建器
     */
    class MulBuilder
            : public builder::OpParamBuilderBase<MulBuilder> {
    public:
        struct Params {
            static constexpr const char *Weight = "weight";
            static constexpr const char *X = "x";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        MulBuilder &weight(T &&value) {
            return set(Params::Weight, value);
        }

        template<typename T>
        MulBuilder &x(T &&value) {
            return set(Params::X, value);
        }

        template<typename T>
        MulBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T weight() {
            return get<T>(Params::Weight);
        }

        template<typename T>
        T x() {
            return get<T>(Params::X);
        }
    };

    /**
     * @brief Reshape算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Reshape;
    /**
     * @brief Reshape算子参数构建器
     */
    class ReshapeBuilder
            : public builder::OpParamBuilderBase<ReshapeBuilder> {
    public:
        struct Params {
            static constexpr const char *EmbdHeadNum = "embd_head_num";
            static constexpr const char *HeadNum = "head_num";
            static constexpr const char *TokenNum = "token_num";
            static constexpr const char *BatchNum = "batch_num";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        ReshapeBuilder &embd_head_num(T &&value) {
            return set(Params::EmbdHeadNum, value);
        }

        template<typename T>
        ReshapeBuilder &head_num(T &&value) {
            return set(Params::HeadNum, value);
        }

        template<typename T>
        ReshapeBuilder &token_num(T &&value) {
            return set(Params::TokenNum, value);
        }

        template<typename T>
        ReshapeBuilder &batch_num(T &&value) {
            return set(Params::BatchNum, value);
        }


        template<typename T>
        ReshapeBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        ReshapeBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T embd_head_num() {
            return get<T>(Params::EmbdHeadNum);
        }

        template<typename T>
        T head_num() {
            return get<T>(Params::HeadNum);
        }

        template<typename T>
        T token_num() {
            return get<T>(Params::TokenNum);
        }

        template<typename T>
        T batch_num() {
            return get<T>(Params::BatchNum);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief Rope算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Rope;
    /**
     * @brief Rope算子参数构建器
     */
    class RopeBuilder
            : public builder::OpParamBuilderBase<RopeBuilder> {
    public:
        struct Params {
            static constexpr const char *RopeType = "rope_type";
            static constexpr const char *RopeTable = "rope_table";
            static constexpr const char *TokenIdx = "token_idx";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        RopeBuilder &rope_type(T &&value) {
            return set(Params::RopeType, value);
        }

        template<typename T>
        RopeBuilder &rope_table(T &&value) {
            return set(Params::RopeTable, value);
        }

        template<typename T>
        RopeBuilder &token_idx(T &&value) {
            return set(Params::TokenIdx, value);
        }

        template<typename T>
        RopeBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        RopeBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T rope_type() {
            return get<T>(Params::RopeType);
        }

        template<typename T>
        T rope_table() {
            return get<T>(Params::RopeTable);
        }

        template<typename T>
        T token_idx() {
            return get<T>(Params::TokenIdx);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief FlashAttn算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class FlashAttn;
    /**
     * @brief FlashAttn算子参数构建器
     */
    class FlashAttnBuilder
            : public builder::OpParamBuilderBase<FlashAttnBuilder> {
    public:
        struct Params {
            static constexpr const char *Q = "q_tensor";
            static constexpr const char *K = "k_tensor";
            static constexpr const char *V = "v_tensor";
            static constexpr const char *Mask = "mask";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        FlashAttnBuilder &q(T &&value) {
            return set(Params::Q, value);
        }

        template<typename T>
        FlashAttnBuilder &k(T &&value) {
            return set(Params::K, value);
        }

        template<typename T>
        FlashAttnBuilder &v(T &&value) {
            return set(Params::V, value);
        }

        template<typename T>
        FlashAttnBuilder &mask(T &&value) {
            return set(Params::Mask, value);
        }

        template<typename T>
        FlashAttnBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T q() {
            return get<T>(Params::Q);
        }

        template<typename T>
        T k() {
            return get<T>(Params::K);
        }

        template<typename T>
        T v() {
            return get<T>(Params::V);
        }

        template<typename T>
        T mask() {
            return get<T>(Params::Mask);
        }
    };

    /**
     * @brief PagedFlashAttn算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class PagedFlashAttn;
    /**
     * @brief PagedFlashAttnRope算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class PagedFlashAttnRope;
    /**
     * @brief PagedFlashAttn算子参数构建器
     */
    class PagedFlashAttnBuilder
            : public builder::OpParamBuilderBase<PagedFlashAttnBuilder> {
    public:
        struct Params {
            static constexpr const char *Q = "q_tensor";
            static constexpr const char *K = "k_tensor";
            static constexpr const char *V = "v_tensor";
            static constexpr const char *Mask = "mask";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        PagedFlashAttnBuilder &q(T &&value) {
            return set(Params::Q, value);
        }

        template<typename T>
        PagedFlashAttnBuilder &k(T &&value) {
            return set(Params::K, value);
        }

        template<typename T>
        PagedFlashAttnBuilder &v(T &&value) {
            return set(Params::V, value);
        }

        template<typename T>
        PagedFlashAttnBuilder &mask(T &&value) {
            return set(Params::Mask, value);
        }

        template<typename T>
        PagedFlashAttnBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T q() {
            return get<T>(Params::Q);
        }

        template<typename T>
        T k() {
            return get<T>(Params::K);
        }

        template<typename T>
        T v() {
            return get<T>(Params::V);
        }

        template<typename T>
        T mask() {
            return get<T>(Params::Mask);
        }

        template<typename T>
        T out() {
            return get<T>(Params::Out);
        }
    };

    /**
     * @brief Add算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Add;
    /**
     * @brief Add算子参数构建器
     */
    class AddBuilder
            : public builder::OpParamBuilderBase<AddBuilder> {
    public:
        struct Params {
            static constexpr const char *X1 = "x1";
            static constexpr const char *X2 = "x2";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        AddBuilder &x1(T &&value) {
            return set(Params::X1, value);
        }

        template<typename T>
        AddBuilder &x2(T &&value) {
            return set(Params::X2, value);
        }

        template<typename T>
        AddBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T x1() {
            return get<T>(Params::X1);
        }

        template<typename T>
        T x2() {
            return get<T>(Params::X2);
        }
    };

    /**
     * @brief Norm算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Norm;
    /**
     * @brief NormW算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class NormW;
    /**
     * @brief Norm算子参数构建器
     */
    class NormBuilder
            : public builder::OpParamBuilderBase<NormBuilder> {
    public:
        struct Params {
            static constexpr const char *NormType = "norm_type";
            static constexpr const char *Epsilon = "epsilon";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        NormBuilder &norm_type(T &&value) {
            return set(Params::NormType, value);
        }

        template<typename T>
        NormBuilder &epsilon(T &&value) {
            return set(Params::Epsilon, value);
        }

        template<typename T>
        NormBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        NormBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T norm_type() {
            return get<T>(Params::NormType);
        }

        template<typename T>
        T epsilon() {
            return get<T>(Params::Epsilon);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }

        template<typename T>
        T out() {
            return get<T>(Params::Out);
        }
    };

    /**
     * @brief MemRef算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class MemRef;
    /**
     * @brief MemRef算子参数构建器
     */
    class MemRefBuilder
            : public builder::OpParamBuilderBase<MemRefBuilder> {
    public:
        struct Params {
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        MemRefBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        MemRefBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T in() {
            return get<T>(Params::In);
        }

        template<typename T>
        T out() {
            return get<T>(Params::Out);
        }
    };

    /**
     * @brief DeQuant算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class DeQuant;
    /**
     * @brief DeQuant算子参数构建器
     */
    class DeQuantBuilder
            : public builder::OpParamBuilderBase<DeQuantBuilder> {
    public:
        struct Params {
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        DeQuantBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        DeQuantBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief Quant算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class Quant;
    /**
     * @brief Quant算子参数构建器
     */
    class QuantBuilder
            : public builder::OpParamBuilderBase<QuantBuilder> {
    public:
        struct Params {
            static constexpr const char *QuantDataType = "quant_data_type";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        QuantBuilder &quant_data_type(T &&value) {
            return set(Params::QuantDataType, value);
        }

        template<typename T>
        QuantBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        QuantBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T quant_data_type() {
            return get<T>(Params::QuantDataType);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }

        template<typename T>
        T out() {
            return get<T>(Params::Out);
        }
    };

    /**
     * @brief QuantAligned算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class QuantAligned;
    /**
     * @brief QuantAligned算子参数构建器
     */
    class QuantAlignedBuilder
            : public builder::OpParamBuilderBase<QuantAlignedBuilder> {
    public:
        struct Params {
            static constexpr const char *QuantDataType = "quant_data_type";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        QuantAlignedBuilder &quant_data_type(T &&value) {
            return set(Params::QuantDataType, value);
        }

        template<typename T>
        QuantAlignedBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        QuantAlignedBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T quant_data_type() {
            return get<T>(Params::QuantDataType);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief QuantReshape算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class QuantReshape;
    /**
     * @brief QuantReshape算子参数构建器
     */
    class QuantReshapeBuilder
            : public builder::OpParamBuilderBase<QuantReshapeBuilder> {
    public:
        struct Params {
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        QuantBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        QuantBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief QuantMatMulReshape算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class QuantMatMulReshape;
    /**
     * @brief QuantMatMulReshape算子参数构建器
     */
    class QuantMatMulReshapeBuilder
            : public builder::OpParamBuilderBase<QuantMatMulReshapeBuilder> {
    public:
        struct Params {
            static constexpr const char *TransType = "trans_type";
            static constexpr const char *A = "a";
            static constexpr const char *B = "b";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        QuantMatMulReshapeBuilder &trans_type(T &&value) {
            return set(Params::TransType, value);
        }

        template<typename T>
        QuantMatMulReshapeBuilder &a(T &&value) {
            return set(Params::A, value);
        }

        template<typename T>
        QuantMatMulReshapeBuilder &b(T &&value) {
            return set(Params::B, value);
        }

        template<typename T>
        QuantMatMulReshapeBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T trans_type() {
            return get<T>(Params::TransType);
        }

        template<typename T>
        T a() {
            return get<T>(Params::A);
        }

        template<typename T>
        T b() {
            return get<T>(Params::B);
        }
    };

    /**
     * @brief QuantMatMul算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class QuantMatMul;
    /**
     * @brief QuantMatMul算子参数构建器
     */
    class QuantMatMulBuilder
            : public builder::OpParamBuilderBase<QuantMatMulBuilder> {
    public:
        struct Params {
            static constexpr const char *Weight = "weight";
            static constexpr const char *X = "x";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        QuantMatMulBuilder &weight(T &&value) {
            return set(Params::Weight, value);
        }

        template<typename T>
        QuantMatMulBuilder &x(T &&value) {
            return set(Params::X, value);
        }

        template<typename T>
        QuantMatMulBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T weight() {
            return get<T>(Params::Weight);
        }

        template<typename T>
        T x() {
            return get<T>(Params::X);
        }
    };

    /**
     * @brief MatMul算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class MatMul;
    /**
     * @brief MatMul算子参数构建器
     */
    class MatMulBuilder
            : public builder::OpParamBuilderBase<MatMulBuilder> {
    public:
        struct Params {
            static constexpr const char *TransType = "trans_type";
            static constexpr const char *A = "a";
            static constexpr const char *B = "b";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        MatMulBuilder &trans_type(T &&value) {
            return set(Params::TransType, value);
        }

        template<typename T>
        MatMulBuilder &a(T &&value) {
            return set(Params::A, value);
        }

        template<typename T>
        MatMulBuilder &b(T &&value) {
            return set(Params::B, value);
        }

        template<typename T>
        MatMulBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T trans_type() {
            return get<T>(Params::TransType);
        }

        template<typename T>
        T a() {
            return get<T>(Params::A);
        }

        template<typename T>
        T b() {
            return get<T>(Params::B);
        }
    };

    /**
     * @brief SetRow算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class SetRow;
    /**
     * @brief SetRow算子参数构建器
     */
    class SetRowBuilder
            : public builder::OpParamBuilderBase<SetRowBuilder> {
    public:
        struct Params {
            static constexpr const char *KVCacheCtx = "kv_cache_ctx";
            static constexpr const char *SeqId = "index";
            static constexpr const char *LayerId = "value";
            static constexpr const char *DataType = "data_type";
            static constexpr const char *TensorType = "tensor_type";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        SetRowBuilder &kv_cache_ctx(T &&value) {
            return set(Params::KVCacheCtx, value);
        }

        template<typename T>
        SetRowBuilder &seq_id(T &&value) {
            return set(Params::SeqId, value);
        }

        template<typename T>
        SetRowBuilder &layer_id(T &&value) {
            return set(Params::LayerId, value);
        }

        template<typename T>
        SetRowBuilder &data_type(T &&value) {
            return set(Params::DataType, value);
        }

        template<typename T>
        SetRowBuilder &tensor_type(T &&value) {
            return set(Params::TensorType, value);
        }

        template<typename T>
        SetRowBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        SetRowBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T kv_cache_ctx() {
            return get<T>(Params::KVCacheCtx);
        }

        template<typename T>
        T seq_id() {
            return get<T>(Params::SeqId);
        }

        template<typename T>
        T layer_id() {
            return get<T>(Params::LayerId);
        }

        template<typename T>
        T data_type() {
            return get<T>(Params::DataType);
        }

        template<typename T>
        T tensor_type() {
            return get<T>(Params::TensorType);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief GetRow算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class GetRow;
    /**
     * @brief GetRow算子参数构建器
     */
    class GetRowBuilder
            : public builder::OpParamBuilderBase<GetRowBuilder> {
    public:
        struct Params {
            static constexpr const char *KVCacheCtx = "kv_cache_ctx";
            static constexpr const char *SeqId = "seq_id";
            static constexpr const char *LayerId = "layer_id";
            static constexpr const char *MaxSeqLen = "max_seq_len";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        GetRowBuilder &kv_cache_ctx(T &&value) {
            return set(Params::KVCacheCtx, value);
        }

        template<typename T>
        GetRowBuilder &seq_id(T &&value) {
            return set(Params::SeqId, value);
        }

        template<typename T>
        GetRowBuilder &layer_id(T &&value) {
            return set(Params::LayerId, value);
        }

        template<typename T>
        GetRowBuilder &max_seq_len(T &&value) {
            return set(Params::MaxSeqLen, value);
        }

        template<typename T>
        GetRowBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        GetRowBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T kv_cache_ctx() {
            return get<T>(Params::KVCacheCtx);
        }

        template<typename T>
        T seq_id() {
            return get<T>(Params::SeqId);
        }

        template<typename T>
        T layer_id() {
            return get<T>(Params::LayerId);
        }

        template<typename T>
        T max_seq_len() {
            return get<T>(Params::MaxSeqLen);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief PreRopeTable算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class PreRopeTable;
    /**
     * @brief PreRopeTable算子参数构建器
     */
    class PreRopeTableBuilder
            : public builder::OpParamBuilderBase<PreRopeTableBuilder> {
        static std::string kSubOpPrefix;

    public:
        struct Params {
            static constexpr const char *MaxSeqLen = "max_seq_len";
            static constexpr const char *HiddenDim = "hidden_dim";
            static constexpr const char *Freqs = "freqs";
            static constexpr const char *RopeTable = "rope_table";
        };

    public:
        template<typename T>
        PreRopeTableBuilder &max_seq_len(T &&value) {
            return set(Params::MaxSeqLen, value);
        }

        template<typename T>
        PreRopeTableBuilder &hidden_dim(T &&value) {
            return set(Params::HiddenDim, value);
        }

        template<typename T>
        PreRopeTableBuilder &freqs(T &&value) {
            return set(Params::Freqs, value);
        }

        template<typename T>
        PreRopeTableBuilder &rope_table(T &&value) {
            return set(Params::RopeTable, value);
        }

    public:
        template<typename T>
        T max_seq_len() {
            return get<T>(Params::MaxSeqLen);
        }

        template<typename T>
        T hidden_dim() {
            return get<T>(Params::HiddenDim);
        }

        template<typename T>
        T freqs() {
            return get<T>(Params::Freqs);
        }

        template<typename T>
        T rope_table() {
            return get<T>(Params::RopeTable);
        }
    };

    /**
     * @brief UnaryOP算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class UnaryOP;
    /**
     * @brief UnaryOP算子参数构建器
     */
    class UnaryOPBuilder
            : public builder::OpParamBuilderBase<UnaryOPBuilder> {
    public:
        struct Params {
            static constexpr const char *UnaryType = "unary_type";
            static constexpr const char *X1 = "x1";
            static constexpr const char *X2 = "x2";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        UnaryOPBuilder &unary_type(T &&value) {
            return set(Params::UnaryType, value);
        }

        template<typename T>
        UnaryOPBuilder &x1(T &&value) {
            return set(Params::X1, value);
        }

        template<typename T>
        UnaryOPBuilder &x2(T &&value) {
            return set(Params::X2, value);
        }

        template<typename T>
        UnaryOPBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T unary_type() {
            return get<T>(Params::UnaryType);
        }

        template<typename T>
        T x1() {
            return get<T>(Params::X1);
        }

        template<typename T>
        T x2() {
            return get<T>(Params::X2);
        }
    };

    /****
     * @brief BinaryOP算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class BinaryOP;
    /**
     * @brief BinaryOP算子参数构建器
     */
    class BinaryOPBuilder
            : public builder::OpParamBuilderBase<BinaryOPBuilder> {
    public:
        struct Params {
            static constexpr const char *BinaryType = "binary_type";
            static constexpr const char *X1 = "x1";
            static constexpr const char *X2 = "x2";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        BinaryOPBuilder &binary_type(T &&value) {
            return set(Params::BinaryType, value);
        }

        template<typename T>
        BinaryOPBuilder &x1(T &&value) {
            return set(Params::X1, value);
        }

        template<typename T>
        BinaryOPBuilder &x2(T &&value) {
            return set(Params::X2, value);
        }

        template<typename T>
        BinaryOPBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T binary_type() {
            return get<T>(Params::BinaryType);
        }

        template<typename T>
        T x1() {
            return get<T>(Params::X1);
        }

        template<typename T>
        T x2() {
            return get<T>(Params::X2);
        }
    };

    /**
     * @brief MaskOP算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class MaskOP;
    /**
     * @brief MaskOP算子参数构建器
     */
    class MaskOPBuilder
            : public builder::OpParamBuilderBase<MaskOPBuilder> {
    public:
        struct Params {
            static constexpr const char *MaskType = "mask_type";
            static constexpr const char *DataType = "data_type";
            static constexpr const char *TokenNum = "token_num";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        MaskOPBuilder &mask_type(T &&value) {
            return set(Params::MaskType, value);
        }

        template<typename T>
        MaskOPBuilder &data_type(T &&value) {
            return set(Params::DataType, value);
        }

        template<typename T>
        MaskOPBuilder &token_num(T &&value) {
            return set(Params::TokenNum, value);
        }

        template<typename T>
        MaskOPBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        MaskOPBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T mask_type() {
            return get<T>(Params::MaskType);
        }

        template<typename T>
        T data_type() {
            return get<T>(Params::DataType);
        }

        template<typename T>
        T token_num() {
            return get<T>(Params::TokenNum);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief ViewOP算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class ViewOP;
    /**
     * @brief ViewOP算子参数构建器
     */
    class ViewOPBuilder
            : public builder::OpParamBuilderBase<ViewOPBuilder> {
    public:
        struct Params {
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        ViewOPBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        ViewOPBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief MemOptOP算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class MemOptOP;
    /**
     * @brief MemOptOP算子参数构建器
     */
    class MemOptOPBuilder
            : public builder::OpParamBuilderBase<MemOptOPBuilder> {
    public:
        struct Params {
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        MemOptOPBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        MemOptOPBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T in() {
            return get<T>(Params::In);
        }

        template<typename T>
        T out() {
            return get<T>(Params::Out);
        }
    };

    /**
     * @brief ConvertOP算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class ConvertOP;
    /**
     * @brief ConvertOP算子参数构建器
     */
    class ConvertOPBuilder
            : public builder::OpParamBuilderBase<ConvertOPBuilder> {
    public:
        struct Params {
            static constexpr const char *ConvertDataType = "convert_data_type";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        ConvertOPBuilder &convert_data_type(T &&value) {
            return set(Params::ConvertDataType, value);
        }

        template<typename T>
        ConvertOPBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        ConvertOPBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T convert_data_type() {
            return get<T>(Params::ConvertDataType);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };

    /**
     * @brief GatherOP算子
     * @tparam T 数据类型
     * @tparam DeviceTag 设备类型
     */
    template<typename T, typename DeviceTag>
    class GatherOP;
    /**
     * @brief GatherOP算子参数构建器
     */
    class GatherOPBuilder
            : public builder::OpParamBuilderBase<GatherOPBuilder> {
    public:
        struct Params {
            static constexpr const char *RowIndex = "row_index";
            static constexpr const char *In = "in";
            static constexpr const char *Out = "out";
        };

    public:
        template<typename T>
        GatherOPBuilder &row_index(T &&value) {
            return set(Params::RowIndex, value);
        }

        template<typename T>
        GatherOPBuilder &in(T &&value) {
            return set(Params::In, value);
        }

        template<typename T>
        GatherOPBuilder &out(T &&value) {
            return set(Params::Out, value);
        }

    public:
        template<typename T>
        T row_index() {
            return get<T>(Params::RowIndex);
        }

        template<typename T>
        T in() {
            return get<T>(Params::In);
        }
    };
}
#endif //TFFINFER_XGEMM_H
