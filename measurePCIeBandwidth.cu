#include <iostream>
#include <random>

int main() {
    size_t size = 1 << 30; // 1 Go
    float *d_a, *h_a;

    // Allocation Pinned (Host) et Device pour max perf
    cudaMallocHost(&h_a, size);
    cudaMalloc(&d_a, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Mesure Host -> Device
    cudaEventRecord(start);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    double bandwidth = (size / (milliseconds / 1000.0)) / 1e9; // Go/s
    std::cout << "PCIe Bandwidth (H2D): " << bandwidth << " GB/s" << std::endl;

    cudaFree(d_a);
    cudaFreeHost(h_a);
}