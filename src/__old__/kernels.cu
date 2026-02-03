#include "kernels.cuh"

constexpr uint blockSize = 256;

template <typename LaunchOnce>
static float time_repeats(cudaStream_t stream, int repeats, LaunchOnce launch_once) {
    if (repeats < 1) repeats = 1;

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < repeats; i++) {
        launch_once();
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Catch async launch/runtime errors
    CUDA_CHECK(cudaGetLastError());

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / static_cast<float>(repeats);
}

// 1) naive
__global__ void add_naive_kernel(uint n, const float* x, float* y) {
    for (uint i = 0; i < n; i++) y[i] = x[i] + y[i];
}

float launch_add_naive(const DeviceArraysFP32& d, cudaStream_t stream, int repeats) {
    return time_repeats(stream, repeats, [&]() {
        add_naive_kernel<<<1, 1, 0, stream>>>(d.n, d.x, d.y);
    });
}

// 2) block
__global__ void add_block_kernel(uint n, const float* x, float* y) {
    uint index = threadIdx.x;
    uint stride = blockDim.x;
    for (uint i = index; i < n; i += stride) y[i] = x[i] + y[i];
}

float launch_add_block(const DeviceArraysFP32& d, cudaStream_t stream, int repeats) {
    return time_repeats(stream, repeats, [&]() {
        add_block_kernel<<<1, blockSize, 0, stream>>>(d.n, d.x, d.y);
    });
}

// 3) grid
__global__ void add_threadBlock_kernel(uint n, const float* x, float* y) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    for (uint i = index; i < n; i += stride) y[i] = x[i] + y[i];
}

float launch_add_threadBlock(const DeviceArraysFP32& d, cudaStream_t stream, int repeats) {
    uint numBlocks = (d.n + blockSize - 1) / blockSize;
    return time_repeats(stream, repeats, [&]() {
        add_threadBlock_kernel<<<numBlocks, blockSize, 0, stream>>>(d.n, d.x, d.y);
    });
}

// 4) bf16 scalar
__global__ void add_threadBlockBF16_kernel(uint n,
                                          const __nv_bfloat16* __restrict__ x,
                                          __nv_bfloat16* __restrict__ y) {
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    for (uint i = index; i < n; i += stride) y[i] = __hadd(x[i], y[i]);
}

float launch_add_threadBlockBF16(const DeviceArraysBF16& d, cudaStream_t stream, int repeats) {
    uint numBlocks = (d.n + blockSize - 1) / blockSize;
    return time_repeats(stream, repeats, [&]() {
        add_threadBlockBF16_kernel<<<numBlocks, blockSize, 0, stream>>>(d.n, d.x, d.y);
    });
}

// 5) bf16 vector (bfloat162)
__global__ void add_threadBlockBF16Vector_kernel(uint n,
                                                const __nv_bfloat16* __restrict__ x,
                                                __nv_bfloat16* __restrict__ y) {
    const auto* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    auto* y2 = reinterpret_cast<__nv_bfloat162*>(y);

    uint n2 = n / 2;
    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for (uint i = index; i < n2; i += stride) {
        y2[i] = __hadd2(x2[i], y2[i]);
    }
}

float launch_add_threadBlockBF16Vector(const DeviceArraysBF16& d, cudaStream_t stream, int repeats) {
    // n doit etre pair
    uint n2 = d.n / 2;
    uint numBlocks = (n2 + blockSize - 1) / blockSize;

    return time_repeats(stream, repeats, [&]() {
        add_threadBlockBF16Vector_kernel<<<numBlocks, blockSize, 0, stream>>>(d.n, d.x, d.y);
    });
}