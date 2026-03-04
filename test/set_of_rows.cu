//
// Created by nkk on 2026/2/7.
//

#include <vector>
#include <random>
#include "cublas_v2.h"
#include "mma.h"
#include "kernel/include/kernel_util.h"
#include <cstdint>
#include <cstring>
using T = half;
template<typename T, const int WARP_SIZE, const int BLOCK_ROW_SIZE, const int BLOCK_COL_SIZE>
__global__ __forceinline__ void get_value_by_table(
                                                   const int64_t *__restrict__ block_table,
                                                   T *__restrict__ dst,
                                                   const int dim0,
                                                   const int dim1,
                                                   const int dim2,
                                                   const int start_row,
                                                   const int start_offset,
                                                   const int token_num) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int head_index = blockIdx.x;
    const int batch_index = blockIdx.z;
    const int block_row_index = blockIdx.y * BLOCK_ROW_SIZE;

    const int thread_x = thread_id % WARP_SIZE;
    const int warp_id = thread_id / WARP_SIZE;
    const int VEC_DIM_ROW = BLOCK_ROW_SIZE / blockDim.y; // 4
    const int VEC_DIM_COL = BLOCK_COL_SIZE / blockDim.x; // 1

    const int src_row_base = start_row + block_row_index;
    const int src_col_base = head_index * dim0 / 2;
    const int g_index = batch_index * dim2 * dim1 * dim0;
    const auto dst_block_ptr = dst + g_index;

#pragma unroll
    for (int i = 0; i < VEC_DIM_ROW; i++) {
        const int row = src_row_base + warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
        if (row >= dim2) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < VEC_DIM_COL; j++) {
            const int col = src_col_base + thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL;
            if (col >= dim1 * dim0) {
                continue;
            }
            const int dst_row_base = warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
            const int dst_row = start_offset + dst_row_base;
            if (dst_row_base < token_num) {
                auto *dst_ptr = reinterpret_cast<half2 *>(dst_block_ptr);
                auto *src_ptr = reinterpret_cast<half2 *>(block_table[row * dim1 + head_index]);
                dst_ptr[dst_row * dim1 * dim0 / 2 + col] = src_ptr[thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL];
                // if (thread_x == 0 && warp_id == 0 && head_index == 1) {
                //     printf("thread_x: %d, warp_id:%d, block_table[%d]: %lld, src_ptr[%d].x: %lf, src_ptr[%d].y: %lf "
                //            "dst[%d].x: %lf, dst[%d].y: %lf\n",
                //         thread_x, warp_id,
                //         row * dim1 + head_index,
                //         block_table[row * dim1 + head_index],
                //         thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL,
                //         __half2float(src_ptr[thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL].x),
                //         col,
                //         __half2float(src_ptr[thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL].y),
                //         dst_row * dim1 * dim0 / 2 + col,
                //         __half2float(dst_ptr[dst_row * dim1 * dim0 / 2 + col].x),
                //         dst_row * dim1 * dim0 / 2 + col,
                //         __half2float(dst_ptr[dst_row * dim1 * dim0 / 2 + col].y));
                // }
            }
        }
    }
}

template<const int BLOCK_SIZE>
__global__ __forceinline__ void get_of_rows_kernel(const half *__restrict__ src,
                                                   int64_t *__restrict__ dst,
                                                   const int src_dim0,
                                                   const int dst_dim0,
                                                   const int dim1,
                                                   const int dim2,
                                                   const int dim3,
                                                   const int start_row,
                                                   const int start_offset,
                                                   const int token_num) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int batch_index = blockIdx.y;
    const int block_index = blockIdx.x;
    const int g_dst_index = batch_index * dim2 * dim1 * dst_dim0 + block_index * BLOCK_SIZE * dst_dim0 + start_offset *
                            dim1 * dst_dim0;
    const int g_src_index = batch_index * dim2 * dim1 * src_dim0 + block_index * BLOCK_SIZE * src_dim0 + start_row *
                            dim1 * src_dim0;

    auto *dst_block_ptr = dst + g_dst_index;
    auto *src_block_ptr = src + g_src_index;
    if ((g_dst_index + thread_id * dst_dim0) >= (start_offset *
                            dim1 * dst_dim0 + dim3 * dim1 * token_num * dst_dim0)) {
        return;
    }
    if ((g_src_index + thread_id * src_dim0) >= (start_row *
                            dim1 * src_dim0 + dim3 * dim1 * dim2 * src_dim0)) {
        return;
    }
    dst_block_ptr[thread_id * dst_dim0] = reinterpret_cast<int64_t>(&src_block_ptr[thread_id * src_dim0]);
    // if (block_index == 0 && start_row == 32) {
    //     printf("thread_id: %d,block_index: %d, g_dst_index: %d, g_src_index : %d src[%d]: %lf, src[%d]: %lld \n",
    //            thread_id, block_index, g_dst_index + thread_id * dst_dim0, g_src_index + thread_id * src_dim0,
    //            thread_id * src_dim0,
    //            __half2float(src_block_ptr[thread_id * src_dim0]),
    //            thread_id * src_dim0,
    //            reinterpret_cast<int64_t>(&src_block_ptr[thread_id * src_dim0]));
    // }
}

template<typename T, const int WARP_SIZE, const int BLOCK_ROW_SIZE, const int BLOCK_COL_SIZE>
__global__ __forceinline__ void set_of_rows_kernel(const float *__restrict__ src, T *__restrict__ dst,
                                                   const int dim0,
                                                   const int dim1,
                                                   const int dim2,
                                                   const int start_row,
                                                   const int start_offset,
                                                   const int token_num) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int head_index = blockIdx.x;
    const int batch_index = blockIdx.z;
    const int block_row_index = blockIdx.y * BLOCK_ROW_SIZE;

    const int thread_x = thread_id % WARP_SIZE;
    const int warp_id = thread_id / WARP_SIZE;
    const int VEC_DIM_ROW = BLOCK_ROW_SIZE / blockDim.y; // 4
    const int VEC_DIM_COL = BLOCK_COL_SIZE / blockDim.x; // 1

    const int src_row_base = start_row + block_row_index;
    const int src_col_base = head_index * dim0 / 2;
    const int g_index = batch_index * dim2 * dim1 * dim0;
    const auto src_block_ptr = src + g_index;
    const auto dst_block_ptr = dst + g_index;
    const auto *src_ptr = reinterpret_cast<const float2 *>(src_block_ptr);

#pragma unroll
    for (int i = 0; i < VEC_DIM_ROW; i++) {
        const int row = src_row_base + warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
        if (row >= dim2) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < VEC_DIM_COL; j++) {
            const int col = src_col_base + thread_x + j * BLOCK_COL_SIZE / VEC_DIM_COL;
            if (col >= dim1 * dim0) {
                continue;
            }
            const int dst_row_base = warp_id + i * BLOCK_ROW_SIZE / VEC_DIM_ROW;
            const int dst_row = start_offset + dst_row_base;
            if (dst_row_base < token_num) {
                if (std::is_same_v<T, float>) {
                    auto *dst_ptr = reinterpret_cast<float2 *>(dst_block_ptr);
                    dst_ptr[dst_row * dim1 * dim0 / 2 + col] = src_ptr[row * dim1 * dim0 / 2 + col];
                } else if (std::is_same_v<T, half>) {
                    auto *dst_ptr = reinterpret_cast<half2 *>(dst_block_ptr);
                    dst_ptr[dst_row * dim1 * dim0 / 2 + col] = __float22half2_rn(src_ptr[row * dim1 * dim0 / 2 + col]);
                }
            }
        }
    }
}

void set_of_rows_cpu(const float *__restrict__ src, T *__restrict__ dst,
                     const int dim0,
                     const int dim1,
                     const int dim2,
                     const int start_row,
                     const int start_offset,
                     const int token_num) {
    const int src_row_base = start_row;
    for (int ss = 0; ss < 32; ++ss) {
        const int row = src_row_base + ss;
        if (row >= dim2) {
            continue;
        }
        for (int hh = 0; hh < dim1; ++hh) {
            const int src_col_base = hh * dim0;
            for (int dd = 0; dd < dim0; ++dd) {
                if ((ss) >= token_num || (src_col_base + dd) >= dim1 * dim0) {
                    continue;
                }
                if (std::is_same_v<T, half>) {
                    dst[(start_offset + ss) * dim1 * dim0 + src_col_base + dd] =
                            __float2half_rn(src[row * dim1 * dim0 + src_col_base + dd]);
                } else if (std::is_same_v<T, float>) {
                    dst[(start_offset + ss) * dim1 * dim0 + src_col_base + dd] =
                            (src[row * dim1 * dim0 + src_col_base + dd]);
                }
            }
        }
    }
}

template<typename T>
void get_of_rows_cpu(const T *__restrict__ src, int64_t *__restrict__ dst,
                     const int src_dim0,
                     const int dst_dim0,
                     const int dim1,
                     const int dim2,
                     const int start_row,
                     const int start_offset,
                     const int token_num) {
    const int src_row_base = start_row;
    for (int ss = 0; ss < 32; ++ss) {
        const int row = src_row_base + ss;
        if (row >= dim2) {
            continue;
        }
        for (int hh = 0; hh < dim1; ++hh) {
            const int src_col_base = hh * src_dim0;
            const int dst_col_base = hh * dst_dim0;
            if ((ss) >= token_num || (src_col_base) >= dim1 * src_dim0) {
                continue;
            }
            dst[(start_offset + ss) * dim1 * dst_dim0 + dst_col_base] =
                    reinterpret_cast<int64_t>(&src[row * dim1 * src_dim0 + src_col_base]);
        }
    }
}

template<typename T, const int PAGE_SIZE>
void set_of_rows(std::vector<float> weight_cpu_result, const int d, const int h, const int s,
                 T *cache_tensor = nullptr) {
    float *gpu_ptr = nullptr;
    cudaMalloc((void **) &gpu_ptr, weight_cpu_result.size() * sizeof(float));
    cudaMemcpy(gpu_ptr, weight_cpu_result.data(), weight_cpu_result.size() * sizeof(float), cudaMemcpyHostToDevice);


    const int cache_s = PAGE_SIZE * ((s + PAGE_SIZE - 1) / PAGE_SIZE);
    if (cache_tensor == nullptr) {
        cudaMalloc((void **) &cache_tensor, d * h * cache_s * sizeof(T));
    }

    std::vector<T> cache_cpu_result_cpu;
    cache_cpu_result_cpu.resize(d * h * cache_s);

    std::vector<T> cache_cpu_result_gpu;
    cache_cpu_result_gpu.resize(d * h * cache_s);
    for (int i = 0; i < s; i += PAGE_SIZE) {
        constexpr int BLOCK_COL_SIZE = 2 * PAGE_SIZE;
        constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;

        int token_num = PAGE_SIZE;
        if ((i + PAGE_SIZE) > s) {
            token_num = s % PAGE_SIZE;
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        dim3 grid(h, (PAGE_SIZE + BLOCK_ROW_SIZE - 1) / BLOCK_ROW_SIZE, 1);
        dim3 block(32, 8);
        set_of_rows_kernel<T, 32, BLOCK_ROW_SIZE, BLOCK_COL_SIZE><<<grid, block>>>(gpu_ptr,
            cache_tensor,
            d, h, s, i, i, token_num);

        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);


        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("Kernel time: %.4f ms\n", milliseconds);

        set_of_rows_cpu(weight_cpu_result.data(), cache_cpu_result_cpu.data(),
                        d, h, s, i, i, token_num);
    }
    CudaSafeCall(cudaMemcpy(cache_cpu_result_gpu.data(), cache_tensor, cache_cpu_result_gpu.size() * sizeof(T),
        cudaMemcpyKind::cudaMemcpyDeviceToHost));


    if constexpr (std::is_same_v<T, half>) {
        for (int i = 0; i < cache_cpu_result_gpu.size(); ++i) {
            float delta = __half2float(cache_cpu_result_gpu[i]) - __half2float(cache_cpu_result_cpu[i]);
            if (delta > 0.001f) {
                printf("i: %d, error: %lf \n", i, delta);
            }
        }
    }


    printf("set_of_rows success\n");
    cudaFree(gpu_ptr);
    gpu_ptr = nullptr;
}

template<typename T, const int PAGE_SIZE>
void get_of_rows(std::vector<float> weight_cpu_result, const int d, const int h, const int s) {
    const int cache_s = PAGE_SIZE * ((s + PAGE_SIZE - 1) / PAGE_SIZE);
    T *cache_tensor = nullptr;
    cudaMalloc((void **) &cache_tensor, d * h * cache_s * sizeof(T));

    set_of_rows<T, PAGE_SIZE>(weight_cpu_result, d, h, s, cache_tensor);


    std::vector<T> cache_cpu_result_cpu;
    cache_cpu_result_cpu.resize(d * h * cache_s);
    cudaMemcpy(cache_cpu_result_cpu.data(), cache_tensor, cache_cpu_result_cpu.size() * sizeof(T),
               cudaMemcpyDeviceToHost);

    int64_t *block_table = nullptr;
    cudaMalloc((void **) &block_table, h * cache_s * sizeof(int64_t));

    T *block_table_value_result = nullptr;
    cudaMalloc((void **) &block_table_value_result, d * h * cache_s * sizeof(T));

    std::vector<int64_t> block_table_cpu;
    block_table_cpu.resize(h * cache_s);

    std::vector<int64_t> block_table_gpu;
    block_table_gpu.resize(h * cache_s);

    for (int i = 0; i < s; i += PAGE_SIZE) {
        constexpr int BLOCK_SIZE = PAGE_SIZE;

        int token_num = PAGE_SIZE;
        if ((i + PAGE_SIZE) > s) {
            token_num = s % PAGE_SIZE;
        }

        dim3 grid((PAGE_SIZE * h + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        dim3 block(PAGE_SIZE, 1);

        get_of_rows_kernel<BLOCK_SIZE><<<grid, block>>>(cache_tensor, block_table,
                                                        d, 1, h, s, 1,
                                                        i, i, token_num);

        get_of_rows_cpu(cache_cpu_result_cpu.data(), block_table_cpu.data(),
                        d, 1, h, s, i, i, token_num);
    }

    for (int i = 0; i < s; i += PAGE_SIZE) {
        constexpr int BLOCK_COL_SIZE = 2 * PAGE_SIZE;
        constexpr int BLOCK_ROW_SIZE = PAGE_SIZE;

        int token_num = PAGE_SIZE;
        if ((i + PAGE_SIZE) > s) {
            token_num = s % PAGE_SIZE;
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        dim3 grid(h, (PAGE_SIZE + BLOCK_ROW_SIZE - 1) / BLOCK_ROW_SIZE, 1);
        dim3 block(32, 8);
        get_value_by_table<T, 32, BLOCK_ROW_SIZE, BLOCK_COL_SIZE><<<grid, block>>>(block_table,
            block_table_value_result,
            d, h, s, i, i, token_num);

        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);


        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        printf("Kernel time: %.4f ms\n", milliseconds);

    }

    cudaMemcpy(block_table_gpu.data(), block_table, block_table_gpu.size() * sizeof(int64_t),
        cudaMemcpyDeviceToHost);

    std::vector<T> cache_cpu_result_gpu;
    cache_cpu_result_gpu.resize(d * h * cache_s);
    cudaMemcpy(cache_cpu_result_gpu.data(), block_table_value_result, cache_cpu_result_gpu.size() * sizeof(T),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < s; ++i) {
        for (int j = 0; j < h; ++j) {
            T *block_table_cpu_result_ptr = (&cache_cpu_result_gpu[i * h * d + j * d]);
            T *cache_cpu_result_ptr = &cache_cpu_result_cpu[i * h * d + j * d];
            for (int k = 0; k < d; ++k) {
                float cpu_value = __half2float(cache_cpu_result_ptr[k]);
                float gpu_value = __half2float(block_table_cpu_result_ptr[k]);
                float delta = cpu_value - gpu_value;
                if (fabs(delta) > 0.001f) {
                    printf("s: %d,h: %d, d: %d, error: %lf cpu_value: %lf, gpu_value: %lf\n",
                        i, j, k, delta, cpu_value, gpu_value);
                    return;
                }
            }
        }
    }
    printf("get_of_rows success\n");
    cudaFree(cache_tensor);
    cache_tensor = nullptr;
    cudaFree(block_table_value_result);
    block_table_value_result = nullptr;
}

int main86578(int argc, char *argv[]) {
    constexpr int PAGE_SIZE = 32;
    std::string filename = "Kcur_normed-0_result.ggml";
    int d = 128;
    int h = 8;
    int s = 36;
    std::vector<float> k_weight_cpu_result;
    k_weight_cpu_result.resize(d * h * s);
    tff::kernel::load_tensor_raw(filename.c_str(), k_weight_cpu_result.data());

    get_of_rows<T, PAGE_SIZE>(k_weight_cpu_result, d, h, s);

    filename = "Vcur-0_result.ggml";
    std::vector<float> v_weight_cpu_result;
    v_weight_cpu_result.resize(d * h * s);
    tff::kernel::load_tensor_raw(filename.c_str(), v_weight_cpu_result.data());
    get_of_rows<T, PAGE_SIZE>(v_weight_cpu_result, d, h, s);
    return 0;
}
