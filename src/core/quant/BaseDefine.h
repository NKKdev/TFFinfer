//
// Created by nkk on 2025/10/28.
//

#ifndef TFFINFER_BASEDEFINE_H
#define TFFINFER_BASEDEFINE_H
#include "device/cuda/cudaInc.h"
#include "utils/util.h"

namespace tff::core::quant {
    using dequantize = std::function<void(
        const void *src,
        float *dst,
        int64_t k
    )>;
    using quantize = std::function<void(
        const float *src,
        void *dst,
        int64_t k
    )>;

    struct Q_8_0 {
        static constexpr int BLOCK_SIZE = 32;
        half d;
        int8_t qs[BLOCK_SIZE];

        static void dequantize(const Q_8_0 *blocks, float *out, const int64_t elem_count) {
            const int nb = elem_count / BLOCK_SIZE;
            for (int i = 0; i < nb; ++i) {
                const float scale = tff::utils::fp16_to_fp32(blocks[i].d);
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    out[i * BLOCK_SIZE + j] = blocks[i].qs[j] * scale;
                }
            }
        }

        //
        static void quantize(const float *src, Q_8_0 *blocks, const int64_t elem_count) {
            const int nb = static_cast<int>(elem_count / BLOCK_SIZE);
            for (int i = 0; i < nb; ++i) {
                const float *x = src + i * BLOCK_SIZE;

                float max_abs = 0.0f;
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    max_abs = std::max(max_abs, std::abs(x[j]));
                }
                if (max_abs == 0.0f) {
                    blocks[i].d = tff::utils::fp32_to_fp16(0.0f);
                    for (int j = 0; j < BLOCK_SIZE; ++j) {
                        blocks[i].qs[j] = 0;
                    }
                    continue;
                }
                const float scale = max_abs / 127.0f;
                const float inv_scale = 1.0f / scale;

                blocks[i].d = tff::utils::fp32_to_fp16(scale);

                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    const float v = x[j] * inv_scale;
                    const int32_t iv = static_cast<int32_t>(std::round(v));
                    blocks[i].qs[j] = static_cast<int8_t>(
                        std::max(-127, std::min(127, iv))
                    );
                }
            }
        }

        static constexpr bool is_quantized() { return true; };
    };

    static_assert(sizeof(tff::core::quant::Q_8_0) == 34);

    //
    struct Q_8_1 {
        static constexpr int BLOCK_SIZE = 32;
        half d;
        half s;
        int8_t qs[BLOCK_SIZE];

        static void dequantize(const Q_8_1 *blocks, float *out, const int64_t elem_count) {
            const int nb = elem_count / BLOCK_SIZE;
            for (int i = 0; i < nb; ++i) {
                const float scale = tff::utils::fp16_to_fp32(blocks[i].d);
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    out[i * BLOCK_SIZE + j] = blocks[i].qs[j] * scale;
                }
            }
        }

        //
        static void quantize(const float *src, Q_8_1 *blocks, const int64_t elem_count) {
            const int nb = static_cast<int>(elem_count / BLOCK_SIZE);
            for (int i = 0; i < nb; ++i) {
                const float *x = src + i * BLOCK_SIZE;

                float max_abs = 0.0f;
                float sum = 0.0f;
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    max_abs = std::max(max_abs, std::abs(x[j]));
                    sum += x[j];
                }
                if (max_abs == 0.0f) {
                    blocks[i].d = tff::utils::fp32_to_fp16(0.0f);
                    for (int j = 0; j < BLOCK_SIZE; ++j) {
                        blocks[i].qs[j] = 0;
                    }
                    continue;
                }
                const float scale = max_abs / 127.0f;
                const float inv_scale = 1.0f / scale;

                blocks[i].d = tff::utils::fp32_to_fp16(scale);
                blocks[i].s = tff::utils::fp32_to_fp16(sum);
                for (int j = 0; j < BLOCK_SIZE; ++j) {
                    const float v = x[j] * inv_scale;
                    const int32_t iv = static_cast<int32_t>(std::round(v));
                    blocks[i].qs[j] = static_cast<int8_t>(
                        std::max(-127, std::min(127, iv))
                    );
                }
            }
        }

        static constexpr bool is_quantized() { return true; };
    };
    static_assert(sizeof(tff::core::quant::Q_8_1) == 36);

    //非量化类型;
    template<typename T, typename = void>
    struct QuantScheme {
        static constexpr bool is_quantized() { return false; };
        static constexpr int BLOCK_SIZE = 1;

        static void dequantize(const T *src, float *dst, const int64_t k) {
            for (int64_t i = 0; i < k; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        }

        static void quantize(const float *src, T *dst, int64_t k) {
            for (int64_t i = 0; i < k; ++i) {
                dst[i] = static_cast<T>(src[i]);
            }
        }
    };

    template<typename QuantType>
    constexpr bool is_quant_block_v = std::is_class_v<QuantType> &&
                                      requires
                                      {
                                          QuantType::BLOCK_SIZE;
                                      } &&
                                      requires
                                      {
                                          (QuantType::BLOCK_SIZE > 1);
                                      };

    //
    template<typename QuantType>
    struct QuantScheme<QuantType, std::enable_if_t<is_quant_block_v<QuantType> > > {
        static_assert(
            requires
            {
                QuantType::BLOCK_SIZE;
                requires std::is_invocable_v<decltype(&QuantType::dequantize), const QuantType *, float *, int64_t>;
                requires std::is_invocable_v<decltype(&QuantType::quantize), const float *, QuantType *, int64_t>;
            },
            "BlockType must define BLOCK_SIZE and static dequantize() and static quantize()"
        );


        static constexpr bool is_quantized() {
            return QuantType::is_quantized();
        }

        static constexpr int block_size = QuantType::BLOCK_SIZE;


        static void dequantize(const QuantType *src, float *dst, int64_t elem_count) {
            QuantType::dequantize(src, dst, elem_count);
        }

        //
        static void quantize(const float *src, QuantType *dst, int64_t elem_count) {
            QuantType::quantize(src, dst, elem_count);
        }
    };

    //
    template<typename T>
    dequantize make_dequantize_wrapper() {
        if constexpr (tff::core::quant::QuantScheme<T>::is_quantized()) {
            return [](const void *src, float *dst, int64_t k) {
                tff::core::quant::QuantScheme<T>::dequantize(
                    static_cast<const T *>(src), dst, k
                );
            };
        } else {
            return [](const void *src, float *dst, int64_t k) {
                const T *s = static_cast<const T *>(src);
                for (int64_t i = 0; i < k; ++i) {
                    dst[i] = static_cast<float>(s[i]);
                }
            };
        }
    }

    //
    template<typename T>
    quantize make_quantize_wrapper() {
        if constexpr (tff::core::quant::QuantScheme<T>::is_quantized()) {
            return [](const float *src, void *dst, int64_t k) {
                tff::core::quant::QuantScheme<T>::quantize(
                    src, static_cast<T *>(dst), k
                );
            };
        } else {
            return [](const float *src, void *dst, int64_t k) {
                T *d = static_cast<T *>(dst);
                for (int64_t i = 0; i < k; ++i) {
                    d[i] = static_cast<T>(src[i]);
                };
            };
        }
    }



}

using Q8_0 = tff::core::quant::Q_8_0;
#endif //TFFINFER_BASEDEFINE_H
