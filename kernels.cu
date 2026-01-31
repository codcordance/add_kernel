#include "kernels.cuh"

constexpr int blockSize = 256; //can be changed, for demo.


/**
 * Naive element-wise vector addition: y = x + y.
 * Run all the additions in the same thread.
 *
 * @param n     Total number of elements in the arrays.
 * @param x     Pointer to input vector (Device memory).
 * @param y     Pointer to input/output vector (Device memory).
 * @warning Input pointers must be 4-byte aligned (standard for float).
 */
__global__
void add_naive_kernel(const int n, const float *x, float *y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

void launch_add_naive(const int n, const DataArrays data) {
    add_naive_kernel<<<1, 1>>>(n, data.x, data.y);
}


/**
 * Slightly less naive element-wise vector addition: y = x + y.
 * Dispatch the additions in the threads of one block.
 *
 * @param n     Total number of elements in the arrays.
 * @param x     Pointer to input vector (Device memory).
 * @param y     Pointer to input/output vector (Device memory).
 * @warning Input pointers must be 4-byte aligned (standard for float).
 */
__global__
void add_block_kernel(const int n, const float *x, float *y)
{
    const unsigned int index = threadIdx.x;
    const unsigned int stride = blockDim.x; //here stride = blockSize
    for (unsigned int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

void launch_add_block(const int n, const float *x, float *y) {
    add_block_kernel<<<1, blockSize>>>(n, x, y);
}


/**
 * Element-wise vector addition: y = x + y.
 * Dispatch the additions in the threads of multiple blocks.
 *
 * @param n     Total number of elements in the arrays.
 * @param x     Pointer to input vector (Device memory).
 * @param y     Pointer to input/output vector (Device memory).
 * @warning Input pointers must be 4-byte aligned (standard for float).
 */
__global__
void add_threadBlock_kernel(const int n, const float *x, float *y)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

void launch_add_threadBlock(const int n, const float *x, float *y) {
    int numBlocks = (n + blockSize - 1) / blockSize; //ensure there are enough blocks for all the threads.
    add_threadBlock_kernel<<<numBlocks, blockSize>>>(n, x, y);
}


/**
 * Element-wise vector addition: y = x + y for BF16 format.
 * Dispatch the additions in the threads of multiple blocks.
 *
 * @param n     Total number of elements in the arrays.
 * @param x     Pointer to BF16 input vector (Device memory).
 * @param y     Pointer to BF16 input/output vector (Device memory).
 * @warning Input pointers must be under the BF16 format.
 */
__global__ void add_threadBlockBF16_kernel(const int n, const __nv_bfloat16 *x, __nv_bfloat16 *y) {
    const unsigned int  index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int  stride = blockDim.x * gridDim.x;

    for (unsigned int i = index; i < n; i += stride)
        y[i] = __hadd(x[i], y[i]);
}

void launch_add_threadBlockBF16(const int n, const __nv_bfloat16 *x, __nv_bfloat16 *y) {
    int numBlocks = (n + blockSize - 1) / blockSize;
    add_threadBlockBF16_kernel<<<numBlocks, blockSize>>>(n, x, y);
}

