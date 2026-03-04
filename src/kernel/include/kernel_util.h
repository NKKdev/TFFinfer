//
// Created by nkk on 2025/11/20.
//

#ifndef TFFINFER_KERNEL_UTIL_H
#define TFFINFER_KERNEL_UTIL_H
#include "device/cuda/cudaInc.h"
#include "global/GlobalDefine.h"
#include "mem/Memory.h"
#include "mem/Tensor.h"
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
        } else if (count == 2 && reinterpret_cast<uintptr_t>(addr) % 4 == 0) {
            const half2 *h = reinterpret_cast<const half2 *>(addr);
            out[0] = h[0].x;
            out[1] = h[0].y;
        } else {
#pragma unroll
            for (int i = 0; i < count; ++i) out[i] = __ldg(&addr[i]);
        }
    }

    __device__ __forceinline__ half2 complex_mul_half2(const half2 &a, const half2 &b) {
        half2 res;
        res.x = __hsub(__hmul(a.x, b.x), __hmul(a.y, b.y));
        res.y = __hadd(__hmul(a.x, b.y), __hmul(a.y, b.x));
        return res;
    }

    static size_t get_tensor_data_size(const int64_t ne[4], const size_t nb[4], tff::core::memory::DataType type) {
        size_t total_bytes = 0;
        for (int i = MAX_TENSOR_DIM - 1; i >= 0; --i) {
            if (ne[i] > 1) {
                total_bytes = nb[i] * ne[i];
                break;
            }
        }
        if (total_bytes == 0) {
            // scalar
            total_bytes = core::memory::type_traits_auto[type]._type_size;
        }
        return total_bytes;
    }
    static void load_tensor_raw(const char *filename, void *data) {
        if (!filename) {
            fprintf(stderr, "%s: filename is null\n", __func__);
            return;
        }

        FILE *fp = fopen(filename, "rb");
        if (!fp) {
            fprintf(stderr, "%s: failed to open %s\n", __func__, filename);
            return;
        }

        // 1. Check magic
        uint32_t magic;
        if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid magic in %s\n", __func__, filename);
            fclose(fp);
            return;
        }

        // 2. Read ne and nb
        int32_t ne32[4], nb32[4];
        if (fread(ne32, sizeof(int32_t), 4, fp) != 4 || fread(nb32, sizeof(int32_t), 4, fp) != 4) {
            fprintf(stderr, "%s: failed to read ne/nb\n", __func__);
            fclose(fp);
            return;
        }

        // Convert to int64_t / size_t
        int64_t ne[4];
        size_t nb[4];
        for (int i = 0; i < 4; ++i) {
            ne[i] = (int64_t) ne32[i];
            nb[i] = (size_t) nb32[i];
        }

        // 3. Read type
        int32_t type_i32;
        if (fread(&type_i32, sizeof(int32_t), 1, fp) != 1) {
            fprintf(stderr, "%s: failed to read type\n", __func__);
            fclose(fp);
            return;
        }
        tff::core::memory::DataType type = (tff::core::memory::DataType) type_i32;
        if (type < 0 || type >= core::memory::DataType::TFF_DATA_TYPE_COUNT) {
            fprintf(stderr, "%s: invalid ggml_type %d\n", __func__, type_i32);
            fclose(fp);
            return;
        }

        // 4. Compute data size and allocate
        size_t data_size = get_tensor_data_size(ne, nb, type);
        if (!data) {
            fprintf(stderr, "%s: failed to allocate %zu bytes for data\n", __func__, data_size);
            fclose(fp);
            return;
        }

        if (fread(data, 1, data_size, fp) != data_size) {
            fprintf(stderr, "%s: failed to read data\n", __func__);
            fclose(fp);
            return;
        }
        fclose(fp);

        // printf("%s: loaded raw tensor from %s (shape=[%ld,%ld,%ld,%ld], type=%d, data_size=%zu)\n", __func__, filename,
        //        ne[0], ne[1], ne[2], ne[3], (int) type, data_size);

        return;
    }
#ifdef _DEBUG
    void varify(const char *filename, std::shared_ptr<core::memory::Tensor> &tensor);
    void load_tensor(const char *filename, std::shared_ptr<core::memory::Tensor> &tensor);
    void save_tensor(const char *filename, std::shared_ptr<core::memory::Tensor> &tensor);
    void varify(std::string &op_name, std::string &filename, std::shared_ptr<core::memory::Tensor> &tensor);
#endif

}
#endif //TFFINFER_KERNEL_UTIL_H
