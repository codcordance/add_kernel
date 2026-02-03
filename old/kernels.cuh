#ifndef ADD_KERNEL_KERNELS_CUH
#define ADD_KERNEL_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <iostream>
#include <cstdlib>

using uint = unsigned int;

inline void cuda_check_impl(const cudaError_t e, const char* expr, const char* file, const int line) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(e)
                  << " | expr: " << expr
                  << " | " << file << ":" << line << "\n";
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) cuda_check_impl((expr), #expr, __FILE__, __LINE__)

template <typename T>
struct DeviceArrays {
    uint n = 0;
    const T* x = nullptr;
    T* y = nullptr;
};

using DeviceArraysFP32 = DeviceArrays<float>;
using DeviceArraysBF16 = DeviceArrays<__nv_bfloat16>;

float launch_add_naive(const DeviceArraysFP32& d, cudaStream_t stream, int repeats);
float launch_add_block(const DeviceArraysFP32& d, cudaStream_t stream, int repeats);
float launch_add_threadBlock(const DeviceArraysFP32& d, cudaStream_t stream, int repeats);
float launch_add_threadBlockBF16(const DeviceArraysBF16& d, cudaStream_t stream, int repeats);
float launch_add_threadBlockBF16Vector(const DeviceArraysBF16& d, cudaStream_t stream, int repeats);

#endif
