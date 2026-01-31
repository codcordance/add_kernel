#include "kernels.cuh"

constexpr int blockSize = 256; //can be changed, for demo.

//ADD NAIVE
__global__ void add_naive_kernel(const int n, const float *x, float *y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

/**
 * Naive element-wise vector addition: y = x + y.
 * Run all the additions in the same thread.
 */
void launch_add_naive(const DataArraysFP32 &data) {
    add_naive_kernel<<<1, 1>>>(data.n, data.x, data.y);
}

//ADD BLOCK
__global__ void add_block_kernel(const int n, const float *x, float *y) {
    const unsigned int index = threadIdx.x;
    const unsigned int stride = blockDim.x; //here stride = blockSize
    for (unsigned int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

/**
 * Slightly less naive element-wise vector addition: y = x + y.
 * Dispatch the additions in the threads of one block.
 */
void launch_add_block(const DataArraysFP32 &data) {
    add_block_kernel<<<1, blockSize>>>(data.n, data.x, data.y);
}

//ADD THREAD BLOCK
__global__ void add_threadBlock_kernel(const int n, const float *x, float *y) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

/**
 * Element-wise vector addition: y = x + y.
 * Dispatch the additions in the threads of multiple blocks.
 */
void launch_add_threadBlock(const DataArraysFP32 &data) {
    int numBlocks = (data.n + blockSize - 1) / blockSize; //ensure there are enough blocks for all the threads.
    add_threadBlock_kernel<<<numBlocks, blockSize>>>(data.n, data.x, data.y);
}

//ADD THREAD BLOCK BF16
__global__ void add_threadBlockBF16_kernel(const int n, const __nv_bfloat16 *x, __nv_bfloat16 *y) {
    const unsigned int  index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int  stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < n; i += stride)
        y[i] = __hadd(x[i], y[i]);
}

/**
 * Element-wise vector addition: y = x + y for BF16 format.
 * Dispatch the additions in the threads of multiple blocks.
 */
void launch_add_threadBlockBF16(const DataArraysBF16 &data) {
    int numBlocks = (data.n + blockSize - 1) / blockSize;
    add_threadBlockBF16_kernel<<<numBlocks, blockSize>>>(data.n, data.x, data.y);
}

