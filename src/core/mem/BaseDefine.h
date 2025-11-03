//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEM_BASEDEFINE_H
#define TFFINFER_MEM_BASEDEFINE_H
#include <array>
#include <functional>
#include "quant/BaseDefine.h"
namespace tff::core::memory {
    enum MemCpyKind {
        TFF_MEM_CPY_TYPE_UNKNOWN = 0,
        TFF_MEM_CPY_TYPE_DEVICE2HOST = 1,
        TFF_MEM_CPY_TYPE_HOST2DEVICE = 2,
        TFF_MEM_CPY_TYPE_DEVICE2DEVICE = 2,
    };

    enum DataType {
        TFF_DATA_TYPE_UNKNOWN = -1,
        TFF_DATA_TYPE_F32 = 0,
        TFF_DATA_TYPE_F16 = 1,
        TFF_DATA_TYPE_Q4_0 = 2,
        TFF_DATA_TYPE_Q4_1 = 3,
        TFF_DATA_TYPE_Q5_0 = 6,
        TFF_DATA_TYPE_Q5_1 = 7,
        TFF_DATA_TYPE_Q8_0 = 8,
        TFF_DATA_TYPE_Q8_1 = 9,
        TFF_DATA_TYPE_Q2_K = 10,
        TFF_DATA_TYPE_Q3_K = 11,
        TFF_DATA_TYPE_Q4_K = 12,
        TFF_DATA_TYPE_Q5_K = 13,
        TFF_DATA_TYPE_Q6_K = 14,
        TFF_DATA_TYPE_Q8_K = 15,
        TFF_DATA_TYPE_IQ2_XXS = 16,
        TFF_DATA_TYPE_IQ2_XS = 17,
        TFF_DATA_TYPE_IQ3_XXS = 18,
        TFF_DATA_TYPE_IQ1_S = 19,
        TFF_DATA_TYPE_IQ4_NL = 20,
        TFF_DATA_TYPE_IQ3_S = 21,
        TFF_DATA_TYPE_IQ2_S = 22,
        TFF_DATA_TYPE_IQ4_XS = 23,
        TFF_DATA_TYPE_I8 = 24,
        TFF_DATA_TYPE_I16 = 25,
        TFF_DATA_TYPE_I32 = 26,
        TFF_DATA_TYPE_I64 = 27,
        TFF_DATA_TYPE_F64 = 28,
        TFF_DATA_TYPE_IQ1_M = 29,
        TFF_DATA_TYPE_BF16 = 30,
        TFF_DATA_TYPE_TQ1_0 = 34,
        TFF_DATA_TYPE_TQ2_0 = 35,
        TFF_DATA_TYPE_MXFP4 = 39,
        TFF_DATA_TYPE_COUNT = 40,
    };
    //
    enum ModelTensorType {
        LLM_TENSOR_TYPE_UNKNOWN = -1,
        LLM_TENSOR_TOKEN_EMBD,
        LLM_TENSOR_TOKEN_EMBD_NORM,
        LLM_TENSOR_TOKEN_TYPES,
        LLM_TENSOR_POS_EMBD,
        LLM_TENSOR_OUTPUT,
        LLM_TENSOR_OUTPUT_NORM,
        LLM_TENSOR_ROPE_FREQS,
        LLM_TENSOR_ROPE_FACTORS_LONG,
        LLM_TENSOR_ROPE_FACTORS_SHORT,
        LLM_TENSOR_ATTN_Q,
        LLM_TENSOR_ATTN_K,
        LLM_TENSOR_ATTN_V,
        LLM_TENSOR_ATTN_QKV,
        LLM_TENSOR_ATTN_OUT,
        LLM_TENSOR_ATTN_NORM,
        LLM_TENSOR_ATTN_NORM_2,
        LLM_TENSOR_ATTN_OUT_NORM,
        LLM_TENSOR_ATTN_POST_NORM,
        LLM_TENSOR_ATTN_ROT_EMBD,
        LLM_TENSOR_ATTN_SINKS,
        LLM_TENSOR_FFN_GATE_INP,
        LLM_TENSOR_FFN_GATE_INP_SHEXP,
        LLM_TENSOR_FFN_NORM,
        LLM_TENSOR_FFN_POST_NORM,
        LLM_TENSOR_FFN_GATE,
        LLM_TENSOR_FFN_DOWN,
        LLM_TENSOR_FFN_UP,
        LLM_TENSOR_FFN_ACT,
        LLM_TENSOR_FFN_DOWN_EXP, // split experts for backward compatibility
        LLM_TENSOR_FFN_GATE_EXP,
        LLM_TENSOR_FFN_UP_EXP,
        LLM_TENSOR_FFN_NORM_EXPS,
        LLM_TENSOR_FFN_DOWN_EXPS, // merged experts
        LLM_TENSOR_FFN_GATE_EXPS,
        LLM_TENSOR_FFN_UP_EXPS,
        LLM_TENSOR_FFN_DOWN_SHEXP,
        LLM_TENSOR_FFN_GATE_SHEXP,
        LLM_TENSOR_FFN_UP_SHEXP,
        LLM_TENSOR_FFN_EXP_PROBS_B,
        LLM_TENSOR_ATTN_Q_NORM,
        LLM_TENSOR_ATTN_K_NORM,
        LLM_TENSOR_LAYER_OUT_NORM,
        LLM_TENSOR_POST_ATTN_NORM,
        LLM_TENSOR_POST_MLP_NORM,
        LLM_TENSOR_PER_LAYER_TOKEN_EMBD, // gemma3n
        LLM_TENSOR_PER_LAYER_MODEL_PROJ, // gemma3n
        LLM_TENSOR_PER_LAYER_INP_GATE, // gemma3n
        LLM_TENSOR_PER_LAYER_PROJ, // gemma3n
        LLM_TENSOR_PER_LAYER_PROJ_NORM, // gemma3n
        LLM_TENSOR_PER_LAYER_POST_NORM, // gemma3n
        LLM_TENSOR_ALTUP_PROJ, // gemma3n
        LLM_TENSOR_ALTUP_UNEMBD_PROJ, // gemma3n
        LLM_TENSOR_ALTUP_CORRECT_COEF, // gemma3n
        LLM_TENSOR_ALTUP_CORRECT_SCALE, // gemma3n
        LLM_TENSOR_ALTUP_PREDICT_COEF, // gemma3n
        LLM_TENSOR_ALTUP_ROUTER, // gemma3n
        LLM_TENSOR_ALTUP_ROUTER_NORM, // gemma3n
        LLM_TENSOR_LAUREL_L, // gemma3n
        LLM_TENSOR_LAUREL_R, // gemma3n
        LLM_TENSOR_LAUREL_POST_NORM, // gemma3n
        LLM_TENSOR_SSM_IN,
        LLM_TENSOR_SSM_CONV1D,
        LLM_TENSOR_SSM_X,
        LLM_TENSOR_SSM_DT,
        LLM_TENSOR_SSM_DT_NORM,
        LLM_TENSOR_SSM_A,
        LLM_TENSOR_SSM_B_NORM,
        LLM_TENSOR_SSM_C_NORM,
        LLM_TENSOR_SSM_D,
        LLM_TENSOR_SSM_NORM,
        LLM_TENSOR_SSM_OUT,
        LLM_TENSOR_TIME_MIX_W0,
        LLM_TENSOR_TIME_MIX_W1,
        LLM_TENSOR_TIME_MIX_W2,
        LLM_TENSOR_TIME_MIX_A0,
        LLM_TENSOR_TIME_MIX_A1,
        LLM_TENSOR_TIME_MIX_A2,
        LLM_TENSOR_TIME_MIX_V0,
        LLM_TENSOR_TIME_MIX_V1,
        LLM_TENSOR_TIME_MIX_V2,
        LLM_TENSOR_TIME_MIX_G1,
        LLM_TENSOR_TIME_MIX_G2,
        LLM_TENSOR_TIME_MIX_K_K,
        LLM_TENSOR_TIME_MIX_K_A,
        LLM_TENSOR_TIME_MIX_R_K,
        LLM_TENSOR_TIME_MIX_LERP_X,
        LLM_TENSOR_TIME_MIX_LERP_W,
        LLM_TENSOR_TIME_MIX_LERP_K,
        LLM_TENSOR_TIME_MIX_LERP_V,
        LLM_TENSOR_TIME_MIX_LERP_R,
        LLM_TENSOR_TIME_MIX_LERP_G,
        LLM_TENSOR_TIME_MIX_LERP_FUSED,
        LLM_TENSOR_TIME_MIX_FIRST,
        LLM_TENSOR_TIME_MIX_DECAY,
        LLM_TENSOR_TIME_MIX_DECAY_W1,
        LLM_TENSOR_TIME_MIX_DECAY_W2,
        LLM_TENSOR_TIME_MIX_KEY,
        LLM_TENSOR_TIME_MIX_VALUE,
        LLM_TENSOR_TIME_MIX_RECEPTANCE,
        LLM_TENSOR_TIME_MIX_GATE,
        LLM_TENSOR_TIME_MIX_LN,
        LLM_TENSOR_TIME_MIX_OUTPUT,
        LLM_TENSOR_CHANNEL_MIX_LERP_K,
        LLM_TENSOR_CHANNEL_MIX_LERP_R,
        LLM_TENSOR_CHANNEL_MIX_KEY,
        LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,
        LLM_TENSOR_CHANNEL_MIX_VALUE,
        LLM_TENSOR_ATTN_Q_A,
        LLM_TENSOR_ATTN_Q_B,
        LLM_TENSOR_ATTN_KV_A_MQA,
        LLM_TENSOR_ATTN_KV_B,
        LLM_TENSOR_ATTN_K_B,
        LLM_TENSOR_ATTN_V_B,
        LLM_TENSOR_ATTN_Q_A_NORM,
        LLM_TENSOR_ATTN_KV_A_NORM,
        LLM_TENSOR_ATTN_SUB_NORM,
        LLM_TENSOR_FFN_SUB_NORM,
        LLM_TENSOR_DEC_ATTN_NORM,
        LLM_TENSOR_DEC_ATTN_Q,
        LLM_TENSOR_DEC_ATTN_K,
        LLM_TENSOR_DEC_ATTN_V,
        LLM_TENSOR_DEC_ATTN_OUT,
        LLM_TENSOR_DEC_ATTN_REL_B,
        LLM_TENSOR_DEC_CROSS_ATTN_NORM,
        LLM_TENSOR_DEC_CROSS_ATTN_Q,
        LLM_TENSOR_DEC_CROSS_ATTN_K,
        LLM_TENSOR_DEC_CROSS_ATTN_V,
        LLM_TENSOR_DEC_CROSS_ATTN_OUT,
        LLM_TENSOR_DEC_CROSS_ATTN_REL_B,
        LLM_TENSOR_DEC_FFN_NORM,
        LLM_TENSOR_DEC_FFN_GATE,
        LLM_TENSOR_DEC_FFN_DOWN,
        LLM_TENSOR_DEC_FFN_UP,
        LLM_TENSOR_DEC_OUTPUT_NORM,
        LLM_TENSOR_ENC_ATTN_NORM,
        LLM_TENSOR_ENC_ATTN_Q,
        LLM_TENSOR_ENC_ATTN_K,
        LLM_TENSOR_ENC_ATTN_V,
        LLM_TENSOR_ENC_ATTN_OUT,
        LLM_TENSOR_ENC_ATTN_REL_B,
        LLM_TENSOR_ENC_FFN_NORM,
        LLM_TENSOR_ENC_FFN_GATE,
        LLM_TENSOR_ENC_FFN_DOWN,
        LLM_TENSOR_ENC_FFN_UP,
        LLM_TENSOR_ENC_OUTPUT_NORM,
        LLM_TENSOR_CLS,
        LLM_TENSOR_CLS_OUT,
        LLM_TENSOR_CONV1D,
        LLM_TENSOR_CONVNEXT_DW,
        LLM_TENSOR_CONVNEXT_NORM,
        LLM_TENSOR_CONVNEXT_PW1,
        LLM_TENSOR_CONVNEXT_PW2,
        LLM_TENSOR_CONVNEXT_GAMMA,
        LLM_TENSOR_POS_NET_CONV1,
        LLM_TENSOR_POS_NET_CONV2,
        LLM_TENSOR_POS_NET_NORM,
        LLM_TENSOR_POS_NET_NORM1,
        LLM_TENSOR_POS_NET_NORM2,
        LLM_TENSOR_POS_NET_ATTN_NORM,
        LLM_TENSOR_POS_NET_ATTN_Q,
        LLM_TENSOR_POS_NET_ATTN_K,
        LLM_TENSOR_POS_NET_ATTN_V,
        LLM_TENSOR_POS_NET_ATTN_OUT,
        LLM_TENSOR_SHORTCONV_CONV,
        LLM_TENSOR_SHORTCONV_INPROJ,
        LLM_TENSOR_SHORTCONV_OUTPROJ,
        LLM_TENSOR_NEXTN_EH_PROJ,
        LLM_TENSOR_NEXTN_EMBED_TOKENS,
        LLM_TENSOR_NEXTN_ENORM,
        LLM_TENSOR_NEXTN_HNORM,
        LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD,
        LLM_TENSOR_NEXTN_SHARED_HEAD_NORM,
    };

    template<DataType T>
    struct tff_data_type_to_cpp {
        using type = void; // 默认无效
    };

#define MAP_DATA_TYPE(gguf_enum, cpp_type) \
template<> struct tff_data_type_to_cpp<gguf_enum> { using type = cpp_type; };

    MAP_DATA_TYPE(TFF_DATA_TYPE_I8, int8_t)

    MAP_DATA_TYPE(TFF_DATA_TYPE_I16, int16_t)

    MAP_DATA_TYPE(TFF_DATA_TYPE_I32, int32_t)

    MAP_DATA_TYPE(TFF_DATA_TYPE_I64, int64_t)

    MAP_DATA_TYPE(TFF_DATA_TYPE_F32, float)

    MAP_DATA_TYPE(TFF_DATA_TYPE_F64, double)


#define TFF_DATA_TYPE_LIST \
TFF_TRAITS(F32,      "f32",      1,              sizeof(float),           false, nullptr, nullptr) \
TFF_TRAITS(I8,       "i8",       1,              sizeof(int8_t),          false, nullptr, nullptr)\
TFF_TRAITS(I16,       "i16",       1,              sizeof(int8_t),          false, nullptr, nullptr)\
TFF_TRAITS(I32,      "i32",      1,              sizeof(float),           false, nullptr, nullptr) \
TFF_TRAITS(I64,       "i64",       1,              sizeof(int8_t),          false, nullptr, nullptr)\
TFF_TRAITS(F64,       "f64",       1,              sizeof(int8_t),          false, nullptr, nullptr)\
TFF_TRAITS(Q8_0,      "q8_0",      QK8_0,        sizeof(tff::core::quant::Q_8_0),          true, nullptr, nullptr)


    using tff_to_float_t = std::function<void(
        const void *src,
        float *dst,
        int64_t k
    )>;
    using tff_from_float_t = std::function<void(
        const float *src,
        void *dst,
        int64_t k
    )>;

    struct TFFTypeTraits {
        const char *_type_name;
        int64_t _blck_size;
        size_t _type_size;
        bool _is_quantized;
        tff_to_float_t dequantize_callback;
        tff_from_float_t quantize_callback;
    };


    static std::array<TFFTypeTraits, TFF_DATA_TYPE_COUNT> make_type_traits() {
        std::array<TFFTypeTraits, TFF_DATA_TYPE_COUNT> traits{};

#define TFF_TRAITS(enum_name, name_str, blk, sz, quant, dequantize_func, quantize_func) \
traits[TFF_DATA_TYPE_##enum_name] = TFFTypeTraits{ name_str, blk, sz, quant, dequantize_func, quantize_func};

        TFF_DATA_TYPE_LIST
        return traits;
    }

    static const auto type_traits_auto = make_type_traits();
}
#endif //TFFINFER_MEM_BASEDEFINE_H
