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

    if (naive)
        benchAllNaive(n, repeats, s);

    bench("addThreadBlock (FP32)",                          n, repeats, s, addThreadBlock);
    bench("addThreadBlockRestrict (FP32)",                  n, repeats, s, addThreadBlockRestrict);

    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
