//
// Created by nkk on 2025/12/29.
//
#include <cmath>
#include <vector>
#include <device/cuda/cudaInc.h>
#include <random>
using T = float;
template<const int VEC_DIM_M, const int VEC_DIM_N, const int BLOCK_DIM_M, const int BLOCK_DIM_N>
__global__ void precompute_rope_table(const int max_seq_len, const int dim, const float log_base, float *out_table) {
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.y;
    const int thread_x = thread_id % 32;
    const int warp_id = thread_id / 32;

    const int start_m = blockIdx.y * BLOCK_DIM_M;
    const int start_n = blockIdx.x * BLOCK_DIM_N;

    auto *out_ptr = reinterpret_cast<float2 *>(out_table);
#pragma unroll
    for (int mm = 0; mm < VEC_DIM_M; ++mm) {
        const int row_index = (start_m + warp_id + mm * (BLOCK_DIM_M / VEC_DIM_M));
        if (row_index >= max_seq_len) {
            continue;
        }
#pragma unroll
        for (int nn = 0; nn < VEC_DIM_N; ++nn) {
            const int col_index = (start_n + thread_x + nn * (BLOCK_DIM_N / VEC_DIM_N));
            if (col_index >= dim / 2) {
                continue;
            }

            const float exponent = -static_cast<float>((2.0f * col_index) / static_cast<float>(dim) * log_base);
            const float value = row_index * exp(exponent);
            out_ptr[row_index * (dim / 2) + col_index].x = cos(value);
            out_ptr[row_index * (dim / 2) + col_index].y = sin(value);
        }
    }

}

std::vector<float> precompute_rope_tables_cpu(int max_seq_len, int dim, float base = 10000.0f) {
    std::vector<float> inv_freq(dim / 2);
    for (int i = 0; i < dim / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(base, float(2 * i) / dim);
    }

    std::vector<float> table(max_seq_len * (dim / 2));
    for (int pos = 0; pos < max_seq_len; ++pos) {
        for (int i = 0; i < dim / 2; ++i) {
            float freq = pos * inv_freq[i];
            table[pos * (dim / 2) + i] = freq;
        }
    }
    return table;
}
template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}
void precompute_rope_tables_impl(int max_seq_len, int dim, float base = 10000.0f) {

    float log_base = std::log(base);
    float *out_table_gpu = nullptr;
    cudaMalloc(&out_table_gpu, max_seq_len * dim * sizeof(float));
    cudaMemset(out_table_gpu, 0, max_seq_len * dim * sizeof(float));

    constexpr int BLOCK_DIM_M = 64;
    constexpr int BLOCK_DIM_N = 32;
    constexpr int VEC_DIM_M = 8;
    constexpr int VEC_DIM_N = 1;
    constexpr int THREAD_BLOCK_DIM = (BLOCK_DIM_M / VEC_DIM_M) * (BLOCK_DIM_N / VEC_DIM_N);

    dim3 grid((dim / 2 + BLOCK_DIM_N - 1) / BLOCK_DIM_N, (max_seq_len + BLOCK_DIM_M - 1) / BLOCK_DIM_M);
    dim3 block(THREAD_BLOCK_DIM, 1);


    precompute_rope_table<VEC_DIM_M, VEC_DIM_N, BLOCK_DIM_M, BLOCK_DIM_N><<<grid, block>>>(max_seq_len, dim, log_base, out_table_gpu);

    std::vector<float> table_gpu_result;
    table_gpu_result.resize(max_seq_len * dim);
    cudaMemcpy(table_gpu_result.data(), out_table_gpu, max_seq_len * dim * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(out_table_gpu);
    out_table_gpu = nullptr;
#ifdef _DEBUG
    auto angles = precompute_rope_tables_cpu(max_seq_len, dim, base);
    std::vector<float> cos_table(angles.size()), sin_table(angles.size());
    for (size_t i = 0; i < angles.size(); ++i) {
        cos_table[i] = std::cos(angles[i]);
        sin_table[i] = std::sin(angles[i]);
    }
    std::vector<float> cos_sin_table(max_seq_len * dim);
    for (int m = 0; m < max_seq_len; ++m) {
        for (int d = 0; d < dim; d += 2) {
            cos_sin_table[m * dim + d] = cos_table[m * dim / 2 + d / 2];
            cos_sin_table[m * dim + d + 1] = sin_table[m * dim / 2 + d / 2];
        }
    }
    for (int m = 0; m < max_seq_len; ++m) {
        for (int d = 0; d < dim / 2; ++d) {
            float cos_delta = cos_sin_table[m * dim + d] - table_gpu_result[m * dim + d];
            float sin_table = cos_sin_table[m * dim + d + 1] - table_gpu_result[m * dim + d + 1];
            if (fabs(cos_delta) > 0.001f || fabs(sin_table) > 0.001f) {
                printf("error m: %d, d: %d,cpu cos: %lf,gpu cos: %lf, cpu sin: %lf, gpu sin: %lf\n",
                    m,d, cos_sin_table[m * dim + d], table_gpu_result[m * dim + d], cos_sin_table[m * dim + d + 1], table_gpu_result[m * dim + d + 1]);
                return;
            }
        }
    }
    printf("success\n");
#endif
}
int main12345(int argc, char **argv) {

    int dim = 1024;
    int m = dim;
    int n = dim;
    int k = 64;

    precompute_rope_tables_impl(std::max(m,n), k);
    return 0;
}
