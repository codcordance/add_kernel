#include <iostream>
#include <cuda_bf16.h>
#include <random>

#include "kernels.cuh"

template <typename T> __host__ T convert_to(float val) {
    if constexpr(std::is_same_v<T, __nv_bfloat16>)
        return __float2bfloat16(val);

    return val;
}

template <typename T> __host__ float convert_from(T val) {
    if constexpr(std::is_same_v<T, __nv_bfloat16>)
        return __bfloat162float(val);

    return static_cast<float>(val);
}

template<typename T>
DataArrays<T> init_arrays(const uint n, const float sum) {
    T *x, *y;
    cudaMallocManaged(&x, n * sizeof(T));
    cudaMallocManaged(&y, n * sizeof(T));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution dis(0.0f, sum);

    for (uint i = 0; i < n; i++) {
        float rand_val = dis(gen);
        x[i] = convert_to<T>(rand_val);
        y[i] = convert_to<T>(sum - rand_val);
    }

    return {n, x, y};
}

template <typename T> float process_arrays(const DataArrays<T>& data, const float sum) {
    cudaDeviceSynchronize();

    float error = 0.0f;
    for (uint i = 0; i < data.n; i++) {
        error += std::abs(convert_from<T>(data.y[i]) - sum);
    }

    cudaFree(data.x);
    cudaFree(data.y);
    return error;
}

int main() {
    constexpr uint n = 1<<25;
    constexpr uint instances = 1;
    std::cout << "Will use arrays of size n = " << n << std::endl;

    std::cout << "Launching " << instances << "x add_naive kernel..." << std::endl;
    float totalError = 0.0f;
    for (uint i = 0; i < instances; i++) {
        const float sum = 3.0f + 4.0f * static_cast<float>(i);
        DataArraysFP32 data = init_arrays<float>(n, sum);
        launch_add_naive(data);
        totalError += process_arrays(data, sum);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    std::cout << "Launching " << instances << "x add_block kernel..." << std::endl;
    totalError = 0.0f;
    for (uint i = 0; i < instances; i++) {
        const float sum = 3.0f + 4.0f * static_cast<float>(i);
        DataArraysFP32 data = init_arrays<float>(n, sum);
        launch_add_block(data);
        totalError += process_arrays(data, sum);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    std::cout << "Launching " << instances << "x add_threadBlock kernel..." << std::endl;
    totalError = 0.0f;
    for (uint i = 0; i < instances; i++) {
        const float sum = 3.0f + 4.0f * static_cast<float>(i);
        DataArraysFP32 data = init_arrays<float>(n, sum);
        launch_add_threadBlock(data);
        totalError += process_arrays(data, sum);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    std::cout << "Launching " << instances << "x add_threadBlockBF16 kernel..." << std::endl;
    totalError = 0.0f;
    for (uint i = 0; i < instances; i++) {
        const float sum = 3.0f + 4.0f * static_cast<float>(i);
        DataArraysBF16 data = init_arrays<__nv_bfloat16>(n, sum);
        launch_add_threadBlockBF16(data);
        totalError += process_arrays(data, sum);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    return 0;
}