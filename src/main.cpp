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

    const bool naive = false;

    if (naive) {
        bench("addNaive (FP32)",                                n, repeats, s, addNaive);
        bench("addNaiveRestrict (FP32)",                        n, repeats, s, addNaiveRestrict);
        bench("addNaiveSizeT (FP32)",                           n, repeats, s, addNaiveSizeT);
        bench("addNaiveSizeTRestrict (FP32)",                   n, repeats, s, addNaiveSizeTRestrict);
        bench("addNaiveFloat2Restrict (FP32)",                  n, repeats, s, addNaiveFloat2Restrict);
        bench("addNaiveFloat4Restrict (FP32)",                  n, repeats, s, addNaiveFloat4Restrict);
        //bench("addNaiveFloat2Restrict (FP32)",                     n, repeats, s, addNaiveFloat2Restrict);
        bench("addNaiveFloat4NoTail (FP32)",                    n, repeats, s, addNaiveFloat4NoTail);
        bench("addNaiveFloat2RestrictNoTail (FP32)",            n, repeats, s, addNaiveFloat2RestrictNoTail);
        bench("addNaiveFloat4RestrictNoTail (FP32)",            n, repeats, s, addNaiveFloat4RestrictNoTail);
    }

    bench("addThreadBlock (FP32)",                          n, repeats, s, addThreadBlock);
    bench("addThreadBlockRestrict (FP32)",                  n, repeats, s, addThreadBlockRestrict);

    CUDA_CHECK(cudaStreamDestroy(s));
    return 0;
}
