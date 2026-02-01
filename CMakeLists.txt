cmake_minimum_required(VERSION 4.1)
project(add_kernel CUDA)

set(CMAKE_CUDA_STANDARD 20)

set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe")
set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler -keep")

add_executable(add_kernel
        main.cu
        kernels.cuh
        kernels.cu
)

add_executable(measure_PCIe_bandwidth
        measurePCIeBandwidth.cu
)

set_target_properties(add_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
CUDA_ARCHITECTURES "native"
)

set_target_properties(measure_PCIe_bandwidth PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
)
