//
// Created by nkk on 2026/1/21.
//
#include <vector>
#include <random>
using T = float;
constexpr int BLOCK_SIZE = 256;
static __device__ __host__ __forceinline__ float op_silu(float x) {
    return x / (1.0f + expf(-x));
}

template<typename T, float (*op)(float)>
__global__ __forceinline__ void op_unary(const T *__restrict__ x1, const T *__restrict__ x2,
    T *result, const int64_t k, const int64_t n1, const int64_t n2) {
    const int64_t g_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (g_index >= k) {
        return;
    }
    const int64_t row = g_index / n1;
    const int64_t col = g_index % n2;
    const int64_t index1 = row * n1 + col;
    const int64_t index2 = row * n2 + col;

    result[g_index] = (T)op(x1[index1]) * x2[index2];
}
template<typename T, float (*op)(float)>
void op_unary_cpu(const T *__restrict__ x1, const T *__restrict__ x2,
    T *result, const int64_t k, const int64_t n1, const int64_t n2) {
    for (int i = 0;i < k; ++i) {
        result[i] = op(x1[i]) * x2[i];
    }
}
template<typename T>
void swiglu(T *x1, T *x2, const int64_t m, const int64_t n1, const int64_t n2) {
    T *x1_gpu = nullptr;
    cudaMalloc((void **)&x1_gpu, m * n1 * sizeof(T));
    cudaMemcpy(x1_gpu, x1, m * n1 * sizeof(T), cudaMemcpyHostToDevice);
    T *x2_gpu = nullptr;
    cudaMalloc((void **)&x2_gpu, m * n2 * sizeof(T));
    cudaMemcpy(x2_gpu, x2, m * n2 * sizeof(T), cudaMemcpyHostToDevice);

    T *result_gpu = nullptr;
    cudaMalloc((void **)&result_gpu, m * n1 * sizeof(T));

    dim3 grid((m * n1 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    op_unary<T,op_silu><<<grid, block>>>(x1_gpu, x2_gpu, result_gpu, m, n1, n2);

    cudaDeviceSynchronize();
    std::vector<T> result;
    result.resize(m * n1);
    cudaMemcpy(result.data(), result_gpu, m * n1 * sizeof(T), cudaMemcpyDeviceToHost);

#ifdef _DEBUG
    std::vector<T> result_cpu(m * n1);
    result_cpu.resize(m * n1);
    op_unary_cpu<T, op_silu>(x1, x2, result_cpu.data(), m, n1, n2);

    for (int i = 0;i < m * n1; ++i) {
        float delta = result_cpu[i] - result[i];
        if (delta > 0.001f) {
            printf("index: %d, error: %lf \n",i , delta);
            return;
        }
    }
    printf("success\n");
#endif

}

template<typename T>
void PopulateVector(std::vector<T> &vector, std::mt19937 &mt, std::uniform_real_distribution<double> &dist) {
    for (auto &element: vector) {
        element = static_cast<T>(dist(mt));
    }
}
int main6757(int argc, char *argv[]) {

    std::mt19937 mt(42);
    std::uniform_real_distribution<double> dist(-4.0, 4.0);

    int dim = 4096;
    int m = dim;
    int n = dim;
    int k = 12288;


    std::vector<T> a_mat;
    a_mat.resize(m * k);
    std::vector<T> b_mat;
    b_mat.resize(n * k);

    swiglu<T>(a_mat.data(), b_mat.data(), m, k, k);

    return 0;
}