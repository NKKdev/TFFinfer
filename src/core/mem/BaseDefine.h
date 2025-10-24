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
        TFF_DATA_TYPE_F32,
        TFF_DATA_TYPE_INT8,
        TFF_DATA_TYPE_INT16,
        TFF_DATA_TYPE_INT32,
        TFF_DATA_TYPE_HALF,
        TFF_DATA_TYPE_DOUBLE,
        TFF_DATA_TYPE_COUNT
    };

#define TFF_DATA_TYPE_LIST \
TFF_TRAITS(F32,      "f32",      1,              sizeof(float),           false, nullptr, nullptr) \
TFF_TRAITS(INT8,       "i8",       1,              sizeof(int8_t),          false, nullptr, nullptr)


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
