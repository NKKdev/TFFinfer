//
// Created by nkk on 2025/10/22.
//

#ifndef TFFINFER_UTIL_H
#define TFFINFER_UTIL_H
#include <cstdint>
#include <math.h>
namespace tff::utils {
#define TFF_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

    //hash
    template<typename T>
    static constexpr void hash_combine(std::size_t &seed, const T &v) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template<typename T, typename... Rest>
    static constexpr void hash_combine(std::size_t &seed, const T &v, const Rest &... rest) {
        hash_combine(seed, v);
        hash_combine(seed, rest...);
    }

    // 用于 pair
    struct pair_hash {
        template<typename T1, typename T2>
        std::size_t operator()(const std::pair<T1, T2> &p) const {
            std::size_t seed = 0;
            hash_combine(seed, p.first, p.second);
            return seed;
        }
    };
    //
    static inline std::string format(const char * fmt, ...) {
        va_list ap;
        va_list ap2;
        va_start(ap, fmt);
        va_copy(ap2, ap);
        int size = vsnprintf(NULL, 0, fmt, ap);
        std::vector<char> buf(size + 1);
        int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);

        va_end(ap2);
        va_end(ap);
        return std::string(buf.data(), size);
    }
    //
    static inline float fp16_to_fp32(uint16_t h) {
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        if (exp == 0x1F) { // Inf or NaN
            return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
        }

        uint32_t f32_bits = (sign << 31) |
                            ((exp == 0 ? 0 : (exp + 127 - 15)) << 23) |
                            (mant << 13);
        return *reinterpret_cast<float*>(&f32_bits);
    }//
    static inline uint16_t fp32_to_fp16(float f) {
        union { float f; uint32_t u; } u = { f };
        uint32_t sign = (u.u >> 16) & 0x8000;
        int32_t exp = ((u.u >> 23) & 0xff) - 127;
        uint32_t mantissa = u.u & 0x7fffff;

        if (exp < -24) {
            return static_cast<uint16_t>(sign);

        } else if (exp < -14) {
            mantissa |= 0x800000;
            mantissa >>= (-14 - exp) + 1;
            return static_cast<uint16_t>(sign | (mantissa >> 13));
        } else if (exp > 15) {
            return static_cast<uint16_t>(sign | 0x7c00 | (mantissa ? 0x200 : 0));
        } else {

            return static_cast<uint16_t>(sign | ((exp + 15) << 10) | (mantissa >> 13));
        }

    }

}
#endif //TFFINFER_UTIL_H
