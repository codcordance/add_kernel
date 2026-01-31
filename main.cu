#include <iostream>
#include <cuda_bf16.h>

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

template <typename T> DataArrays<T> init_arrays(const int n) {
    T *x, *y;
    cudaMallocManaged(&x, n * sizeof(T));
    cudaMallocManaged(&y, n * sizeof(T));

    for (int i = 0; i < n; i++) {
        x[i] = convert_to<T>(1.0f);
        y[i] = convert_to<T>(2.0f);
    }

    return {n, x, y};
}

template <typename T> float process_arrays(const DataArrays<T>& data) {
    cudaDeviceSynchronize();

    float error = 0.0f;
    for (int i = 0; i < data.n; i++) {
        error += std::abs(convert_from<T>(data.y[i]) - 3.0f);
    }

    cudaFree(data.x);
    cudaFree(data.y);
    return error;
}

int main() {
    constexpr int n = 1<<20;
    std::cout << "Will use arrays of size n = " << n << std::endl;

    std::cout << "Launching 10x add_naive kernel..." << std::endl;
    float totalError = 0.0f;
    for (int i = 0; i < 10; i++) {
        DataArraysFP32 data = init_arrays<float>(n);
        launch_add_naive(data);
        totalError += process_arrays(data);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    std::cout << "Launching 10x add_block kernel..." << std::endl;
    totalError = 0.0f;
    for (int i = 0; i < 10; i++) {
        DataArraysFP32 data = init_arrays<float>(n);
        launch_add_block(data);
        totalError += process_arrays(data);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    std::cout << "Launching 10x add_threadBlock kernel..." << std::endl;
    totalError = 0.0f;
    for (int i = 0; i < 10; i++) {
        DataArraysFP32 data = init_arrays<float>(n);
        launch_add_threadBlock(data);
        totalError += process_arrays(data);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    std::cout << "Launching 10x add_threadBlockBF16 kernel..." << std::endl;
    totalError = 0.0f;
    for (int i = 0; i < 10; i++) {
        DataArraysBF16 data = init_arrays<__nv_bfloat16>(n);
        launch_add_threadBlockBF16(data);
        totalError += process_arrays(data);
    }
    std::cout << ">> Total error: " << totalError << std::endl;

    return 0;
}