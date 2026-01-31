#ifndef ADD_KERNEL_KERNELS_CUH
#define ADD_KERNEL_KERNELS_CUH

#include <cuda_bf16.h>

struct DataArrays
{
    float *x;
    float *y;
};

struct DataArraysBF16
{
    __nv_bfloat16 *x;
    __nv_bfloat16 *y;
};

void launch_add_naive(int n, DataArrays data);
void launch_add_block(int n, DataArrays data);
void launch_add_threadBlock(int n, DataArrays data);
void launch_add_threadBlockBF16(int n, DataArraysBF16 data);

#endif //ADD_KERNEL_KERNELS_CUH