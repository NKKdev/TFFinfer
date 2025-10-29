//
// Created by nkk on 2025/10/22.
//

#ifndef TFFINFER_UTIL_H
#define TFFINFER_UTIL_H

namespace tff::utils {
#define TFF_PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))

    //hash
    template<typename T>
    static void hash_combine(std::size_t &seed, const T &v) {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template<typename T, typename... Rest>
    static void hash_combine(std::size_t &seed, const T &v, const Rest &... rest) {
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
}
#endif //TFFINFER_UTIL_H
