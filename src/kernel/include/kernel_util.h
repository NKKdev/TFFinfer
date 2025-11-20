//
// Created by nkk on 2025/11/20.
//

#ifndef TFFINFER_KERNEL_UTIL_H
#define TFFINFER_KERNEL_UTIL_H
#include "device/cuda/cudaInc.h"
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
}
#endif //TFFINFER_KERNEL_UTIL_H