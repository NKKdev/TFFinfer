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
    static void varify(std::string &filename, std::shared_ptr<core::memory::Tensor> &tensor) {
        switch (tensor->get_data_type()) {
            case core::memory::DataType::TFF_DATA_TYPE_F32: {
                std::vector<float> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<float> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                float error_raio = 0.0f;
                for (int b = 0; b < tensor->get_shape()[3]; b++) {
                    for (int s = 0; s < tensor->get_shape()[2]; s++) {
                        float *gpu_ptr = weight_gpu_result.data() + b * tensor->get_shape()[2] * tensor->get_shape()[1] * tensor->get_shape()[0]
                        + s * tensor->get_shape()[1] * tensor->get_shape()[0];
                        float *cpu_ptr = weight_cpu_result.data() + b * tensor->get_shape()[2] * tensor->get_shape()[1] * tensor->get_shape()[0]
                        + s * tensor->get_shape()[1] * tensor->get_shape()[0];
                        for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                            for (int nn = 0; nn < tensor->get_shape()[0]; nn++) {
                                float delta = gpu_ptr[mm * tensor->get_shape()[0] + nn] - cpu_ptr[
                                                  mm * tensor->get_shape()[0] + nn];
                                if (fabs(delta) > 0.01f) {
                                    // tff::log::Logger::error("filename: %s, error: m: %d n: %d, delta: %lf, gpu: %lf, cpu: %lf",
                                    //     filename.c_str(),mm, nn, delta, gpu_ptr[mm * tensor->get_shape()[0] + nn],
                                    //     cpu_ptr[mm * tensor->get_shape()[0] + nn]);

                                    error_raio++;
                                }
                            }
                        }
                    }
                }

                error_raio /= tensor->get_shape()[3] * tensor->get_shape()[2] * tensor->get_shape()[1] * tensor->get_shape()[0];
                if (error_raio > 0.01) {
                    tff::log::Logger::error("error_raio: %lf", error_raio);
                    return;
                }
                break;
            }
            case core::memory::DataType::TFF_DATA_TYPE_Q8_0_ALIGNED: {
                std::vector<Q8_0> weight_cpu_result;
                weight_cpu_result.resize(
                    tensor->get_shape()[0] / Q8_0::BLOCK_SIZE * tensor->get_shape()[1] * tensor->get_shape()[2] *
                    tensor->get_shape()[3]);
                load_tensor_raw(filename.c_str(), weight_cpu_result.data());

                std::vector<Q8_0_ALIGNED> weight_gpu_result;
                weight_gpu_result.resize(weight_cpu_result.size());
                tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), weight_gpu_result.data(),
                                                 tensor->get_bytes(), core::memory::TFF_MEM_CPY_TYPE_DEVICE2HOST);

                for (int mm = 0; mm < tensor->get_shape()[1]; mm++) {
                    for (int nn = 0; nn < tensor->get_shape()[0] / Q8_0::BLOCK_SIZE; nn++) {
                        float delta = weight_gpu_result[mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d -
                                      __half2float(weight_cpu_result[
                                          mm * tensor->get_shape()[0] / Q8_0::BLOCK_SIZE + nn].d);
                        if (fabs(delta) > 0.001f) {
                            tff::log::Logger::error("filename: %s, error: m: %d n: %d, delta: %lf", filename.c_str(),
                                                    mm, nn, delta);
                            //throw std::runtime_error("error");
                            return;
                        }
                    }
                }
                break;
            }
            default:
                break;
        }

        tff::log::Logger::info("layer node op varify (%s) success!", filename.c_str());
    }
#endif
    // static void save_tensor(char * filename, std::shared_ptr<tff::core::memory::Tensor> &tensor) {
    //     FILE * fp = fopen(filename, "wb");
    //     if (!fp) {
    //         fprintf(stderr, "%s: failed to open %s for writing\n", __func__, filename);
    //         return;
    //     }
    //
    //     const uint32_t magic = 0x67676d6c;  // 'ggml' in hex
    //     fwrite(&magic, sizeof(magic), 1, fp);
    //
    //     const int ne0   = tensor->get_shape()[0];
    //     const int ne1   = tensor->get_shape()[1];
    //     const int ne2   = tensor->get_shape()[2];
    //     const int ne3   = tensor->get_shape()[3];
    //     int32_t   ne[4] = { ne0, ne1, ne2, ne3 };
    //     fwrite(ne, sizeof(int32_t), 4, fp);
    //
    //     const int nb0   = tensor->get_strides()[0];
    //     const int nb1   = tensor->get_strides()[1];
    //     const int nb2   = tensor->get_strides()[2];
    //     const int nb3   = tensor->get_strides()[3];
    //     int32_t   nb[4] = { nb0, nb1, nb2, nb3 };
    //     fwrite(nb, sizeof(int32_t), 4, fp);
    //
    //     int32_t type_i32 = (int32_t) tensor->get_data_type();
    //     fwrite(&type_i32, sizeof(int32_t), 1, fp);
    //
    //     size_t total_bytes = 0;
    //     for (int i = 3; i >= 0; --i) {
    //         if (ne[i] > 1) {
    //             total_bytes = nb[i] * ne[i];
    //             break;
    //         }
    //     }
    //     if (total_bytes == 0) {
    //         // scalar
    //         total_bytes = tensor->get_bytes();
    //     }
    //     void *data = malloc(total_bytes);
    //     tensor->get_allocator()->memcopy(tensor->get_buffer()->ptr(), data, total_bytes);
    //     fwrite(data, 1, total_bytes, fp);
    //     free(data);data = nullptr;
    //     fclose(fp);
    //     printf("%s: saved tensor '%s' to %s (%zu bytes)\n", __func__, filename, filename, total_bytes);
    // }

}
#endif //TFFINFER_KERNEL_UTIL_H
