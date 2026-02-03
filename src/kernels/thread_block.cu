#include "thread_block.cuh"

__global__ void addThreadBlockKernel(const int n, const float* x, float* y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

__global__ void addThreadBlockRestrictKernel(const int n, const float* __restrict__ x, float* __restrict__ y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

float addThreadBlock(const data::DeviceData<float> &d, cudaStream_t s) {
    const int n = static_cast<int>(d.n);
    const int numBlocks = (n + blockSize - 1) / blockSize;

    return timeKernel(s, [&] {
        addThreadBlockKernel<<<numBlocks, blockSize, 0, s>>>(n, d.dx, d.dy);
    });
}

float addThreadBlockRestrict(const data::DeviceData<float> &d, cudaStream_t s) {
    const int n = static_cast<int>(d.n);
    const int numBlocks = (n + blockSize - 1) / blockSize;

    return timeKernel(s, [&] {
        addThreadBlockRestrictKernel<<<numBlocks, blockSize, 0, s>>>(n, d.dx, d.dy);
    });
}
