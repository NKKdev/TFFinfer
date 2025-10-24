//
// Created by nkk on 2025/10/22.
//

#ifndef TFFINFER_GGUFDEF_H
#define TFFINFER_GGUFDEF_H

namespace tff::core::model {
#define GGUF_MAGIC   "GGUF"
#define GGUF_VERSION 3
#define GGUF_KEY_GENERAL_ALIGNMENT "general.alignment"
#define GGUF_DEFAULT_ALIGNMENT 32

    enum GGUFType {
        TFF_GGUF_TYPE_UNKNOWN = -1,
        TFF_GGUF_TYPE_UINT8 = 0,
        TFF_GGUF_TYPE_INT8 = 1,
        TFF_GGUF_TYPE_UINT16 = 2,
        TFF_GGUF_TYPE_INT16 = 3,
        TFF_GGUF_TYPE_UINT32 = 4,
        TFF_GGUF_TYPE_INT32 = 5,
        TFF_GGUF_TYPE_FLOAT32 = 6,
        TFF_GGUF_TYPE_BOOL = 7,
        TFF_GGUF_TYPE_STRING = 8,
        TFF_GGUF_TYPE_ARRAY = 9,
        TFF_GGUF_TYPE_UINT64 = 10,
        TFF_GGUF_TYPE_INT64 = 11,
        TFF_GGUF_TYPE_FLOAT64 = 12,
        TFF_GGUF_TYPE_COUNT, // marks the end of the enum
    };

    template<GGUFType T>
    struct gguf_type_to_cpp {
        using type = void; // 默认无效
    };

#define MAP_GGUF_TYPE(gguf_enum, cpp_type) \
template<> struct gguf_type_to_cpp<gguf_enum> { using type = cpp_type; };

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_UINT8, uint8_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_INT8, int8_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_UINT16, uint16_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_INT16, int16_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_UINT32, uint32_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_INT32, int32_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_UINT64, uint64_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_INT64, int64_t)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_FLOAT32, float)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_FLOAT64, double)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_BOOL, bool)

    MAP_GGUF_TYPE(TFF_GGUF_TYPE_STRING, std::string)

#define TFF_GGUF_TYPES \
GGUF_TYPE(TFF_GGUF_TYPE_UINT8,   uint8_t)   \
GGUF_TYPE(TFF_GGUF_TYPE_INT8,    int8_t)    \
GGUF_TYPE(TFF_GGUF_TYPE_UINT16,  uint16_t)  \
GGUF_TYPE(TFF_GGUF_TYPE_INT16,   int16_t)   \
GGUF_TYPE(TFF_GGUF_TYPE_UINT32,  uint32_t)  \
GGUF_TYPE(TFF_GGUF_TYPE_INT32,   int32_t)   \
GGUF_TYPE(TFF_GGUF_TYPE_UINT64,  uint64_t)  \
GGUF_TYPE(TFF_GGUF_TYPE_INT64,   int64_t)   \
GGUF_TYPE(TFF_GGUF_TYPE_FLOAT32, float)     \
GGUF_TYPE(TFF_GGUF_TYPE_FLOAT64, double)    \
GGUF_TYPE(TFF_GGUF_TYPE_BOOL,    bool)      \
GGUF_TYPE(TFF_GGUF_TYPE_STRING,  std::string)
}
#endif //TFFINFER_GGUFDEF_H
