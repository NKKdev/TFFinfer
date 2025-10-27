//
// Created by nkk on 2025/9/28.
//

#ifndef TFFINFER_MEM_BASEDEFINE_H
#define TFFINFER_MEM_BASEDEFINE_H
#include <array>
#include <functional>

namespace tff::core::memory {
    enum MemCpyKind {
        TFF_MEM_CPY_TYPE_UNKNOWN = 0,
        TFF_MEM_CPY_TYPE_GPU2CPU = 1,
        TFF_MEM_CPY_TYPE_CPU2GPU = 2,
    };

    enum DataType {
        TFF_DATA_TYPE_UNKNOWN,
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
TFF_TRAITS(F64,       "f64",       1,              sizeof(int8_t),          false, nullptr, nullptr)


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
        tff_to_float_t _to_float;
        tff_from_float_t _from_float;
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
