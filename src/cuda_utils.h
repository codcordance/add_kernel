#pragma once

#include <iostream>
#include <cuda_runtime.h>

inline void cuda_check_impl(const cudaError_t e, const char* expr, const char* file, const int line) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e)
                  << " | expr: " << expr
                  << " | " << file << ":" << line << "\n";
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)

template <class F>
requires std::invocable<F&>
float timeKernel(cudaStream_t s, F&& launcher) {

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, s));
    std::forward<F>(launcher)();
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Catch async launch/runtime errors
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

constexpr int blockSize = 256;