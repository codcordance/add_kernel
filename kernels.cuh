#ifndef ADD_KERNEL_KERNELS_CUH
#define ADD_KERNEL_KERNELS_CUH

#include <cuda_bf16.h>

template <typename T>
struct DataArrays {
    const int n;
    T *x;
    T *y;
};

using DataArraysFP32 = DataArrays<float>;
using DataArraysBF16 = DataArrays<__nv_bfloat16>;

void launch_add_naive(const DataArraysFP32 &data);
void launch_add_block(const DataArraysFP32 &data);
void launch_add_threadBlock(const DataArraysFP32 &data);
void launch_add_threadBlockBF16(const DataArraysBF16 &data);

#endif //ADD_KERNEL_KERNELS_CUH