#include "bench.h"

#include <iostream>
#include <iomanip>
#include <cstring>

constexpr int Wname = 40;
constexpr int Wms   = 10;
constexpr int Wbw   = 10;
constexpr int Werr  = 14;

void printBench(const char* name, const float timeMsMean, const double bandwidthGbs, const long double errCum) {
    std::cout << std::left  << std::setw(Wname) << name
              << "  " << std::right << std::setw(Wms) << std::fixed << std::setprecision(3) << timeMsMean << " ms"
              << "  | " << std::right << std::setw(Wbw) << std::fixed << std::setprecision(2) << bandwidthGbs << " GB/s"
              << "  | " << std::right << std::setw(Werr) << std::scientific << std::setprecision(3) << errCum
              << std::defaultfloat << "\n";
}

void measurePCIeBandwidth() {
    constexpr size_t bytes = 4ULL << 30; // 4 GiB
    float* h_a = nullptr;
    float* d_a = nullptr;

    CUDA_CHECK(cudaMallocHost(&h_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_a, bytes));

    std::memset(h_a, 0, bytes);
    CUDA_CHECK(cudaMemset(d_a, 1, bytes));

    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_a, h_a, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_a, d_a, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    const long double gbps = (2.0L * static_cast<long double>(bytes)) / (static_cast<long double>(ms) * 1e6L);
    std::cout << "PCIe Bandwidth: " << static_cast<double>(gbps) << " GB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFreeHost(h_a));
}
