#include <iostream>

#include "data.h"
#include "bench.h"
#include "kernels/naive.cuh"
#include "kernels/thread_block.cuh"

int main() {
    measurePCIeBandwidth();

    constexpr size_t n = 1u << 27;
    constexpr int repeats = 4;

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    const bool naive = true;

    if (naive) {
        std::cout << "Benching naive kernels..." << std::endl;
        benchAllNaive(n, repeats, s);
    }

    std::cout << "Benching threadBlock kernels : blockSize = 256, gridStrideBlocks = 2048" << std::endl;
    benchAllThreadBlock(n, repeats, s);

    std::cout << "Benching threadBlock kernels : blockSize = 384, gridStrideBlocks = 2048" << std::endl;
    benchAllThreadBlock(n, repeats, s, 384);

    std::cout << "Benching threadBlock kernels : blockSize = 256, gridStrideBlocks = 768" << std::endl;
    benchAllThreadBlock(n, repeats, s, 256, 768);

    std::cout << "Benching threadBlock kernels : blockSize = 384, gridStrideBlocks = 768" << std::endl;
    benchAllThreadBlock(n, repeats, s, 384, 768);

    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
