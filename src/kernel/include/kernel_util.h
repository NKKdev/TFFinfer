//
// Created by nkk on 2025/11/20.
//

#ifndef TFFINFER_KERNEL_UTIL_H
#define TFFINFER_KERNEL_UTIL_H

namespace tff::kernel {

    __device__ __forceinline__ uint32_t div_u32(uint32_t n, uint32_t magic, int shift) {
        return (__umulhi(n, magic) + n) >> shift;
    }

    __device__ __forceinline__ int32_t div_s32(int32_t n, int32_t magic, int shift) {
        return (__mulhi(n, magic) + n) >> shift;
    }

    __device__ __forceinline__ uint64_t div_u64(uint64_t n, uint64_t magic, int shift) {
        return (__umul64hi(n, magic) + n) >> shift;
    }

    __device__ __forceinline__ int64_t div_s64(int64_t n, int64_t magic, int shift) {
        return (__mul64hi(n, magic) + n) >> shift;
    }

    template<typename T>
    __device__ __forceinline__ void load_vec(const T *addr, T *out, int count);

    template<>
    __device__ __forceinline__ void load_vec<float>(const float *addr, float *out, int count) {
        if (count >= 4 && reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
            float4 v = *reinterpret_cast<const float4 *>(addr);
            out[0] = v.x;
            out[1] = v.y;
            out[2] = v.z;
            out[3] = v.w;
        } else {
#pragma unroll
            for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
        }
    }

    template<>
    __device__ __forceinline__ void load_vec<half>(const half *addr, half *out, int count) {
        if (count == 8 && reinterpret_cast<uintptr_t>(addr) % 16 == 0) {
            uint4 v = *reinterpret_cast<const uint4 *>(addr);
            const half *h = reinterpret_cast<const half *>(&v);
#pragma unroll
            for (int i = 0; i < count; ++i) out[i] = h[i];
        }
        else if (count == 2 && reinterpret_cast<uintptr_t>(addr) % 4 == 0) {
            const half2 *h = reinterpret_cast<const half2 *>(addr);
            out[0] = h[0].x;
            out[1] = h[0].y;
        }
        else {
#pragma unroll
            for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
        }
    }

    __device__ __forceinline__ half2 complex_mul_half2(const half2& a, const half2& b) {
        half2 res;
        res.x = __hsub(__hmul(a.x, b.x), __hmul(a.y, b.y));
        res.y = __hadd(__hmul(a.x, b.y), __hmul(a.y, b.x));
        return res;
    }
}
#endif //TFFINFER_KERNEL_UTIL_H
