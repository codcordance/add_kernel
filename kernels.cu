#include "kernels.cuh"

constexpr uint blockSize = 256; //can be changed, for demo.

//ADD NAIVE
__global__ void add_naive_kernel(const uint n, const float *x, float *y) {
    for (uint i = 0; i < n; i++)
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
__global__ void add_block_kernel(const uint n, const float *x, float *y) {
    const uint index = threadIdx.x;
    const uint stride = blockDim.x; //here stride = blockSize
    for (uint i = index; i < n; i += stride)
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
__global__ void add_threadBlock_kernel(const uint n, const float *x, float *y) {
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint stride = blockDim.x * gridDim.x;
    for (uint i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

/**
 * Element-wise vector addition: y = x + y.
 * Dispatch the additions in the threads of multiple blocks.
 */
void launch_add_threadBlock(const DataArraysFP32 &data) {
    uint numBlocks = (data.n + blockSize - 1) / blockSize; //ensure there are enough blocks for all the threads.
    add_threadBlock_kernel<<<numBlocks, blockSize>>>(data.n, data.x, data.y);
}

//ADD THREAD BLOCK BF16
__global__ void add_threadBlockBF16_kernel(const uint n, const __nv_bfloat16* __restrict__  x, __nv_bfloat16* __restrict__ y) {
     const uint index = blockIdx.x * blockDim.x + threadIdx.x;
     const uint stride = blockDim.x * gridDim.x;

    for (uint i = index; i < n; i += stride)
        y[i] = __hadd(x[i], y[i]);
}

/**
 * Element-wise vector addition: y = x + y for BF16 format.
 * Dispatch the additions in the threads of multiple blocks.
 */
void launch_add_threadBlockBF16(const DataArraysBF16 &data) {
    uint numBlocks = (data.n + blockSize - 1) / blockSize;
    add_threadBlockBF16_kernel<<<numBlocks, blockSize>>>(data.n, data.x, data.y);
}

//ADD THREAD BLOCK BF16 VECTORIZED
__global__ void add_threadBlockBF16Vector_kernel(const uint n, __nv_bfloat16 *x, __nv_bfloat16 *y) {
    //Warning: did not check here since n = 2^something but n must be even !!!
    const auto *x_vec = reinterpret_cast< __nv_bfloat162*>(x);
    auto *y_vec = reinterpret_cast<__nv_bfloat162*>(y);

    // stride and index are divided by 2 since we're working with 2 elements at a time.
    const uint n_vec = n / 2;
    const uint index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint stride = blockDim.x * gridDim.x;

    for (uint i = index; i < n_vec; i += stride) {
        y_vec[i] = __hadd2(x_vec[i], y_vec[i]);
    }
}

/**
 * Element-wise vector addition: y = x + y for BF16 format.
 * This one works with vectorized __nv_bfloat162 = 2 BF16 (4 bytes).
 * Dispatch the additions in the threads of multiple blocks.
 */
void launch_add_threadBlockBF16Vector(const DataArraysBF16 &data) {
    uint numBlocks = (data.n + blockSize - 1) / blockSize;
    add_threadBlockBF16Vector_kernel<<<numBlocks, blockSize>>>(data.n, data.x, data.y);
}
