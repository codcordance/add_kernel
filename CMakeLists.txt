cmake_minimum_required(VERSION 4.1)
project(add_kernel CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/bin/nvcc.exe")
set(CMAKE_CUDA_FLAGS "-keep -O3 -Xptxas -v")

find_package(CUDAToolkit REQUIRED)

add_executable(add_kernel
        src/main.cpp
        src/bench.cpp
        src/bench.h
        src/data.h
        src/cuda_utils.h
        src/kernels/thread_block.cu
        src/kernels/thread_block.cuh
        src/kernels/naive.cu
        src/kernels/naive.cuh
)

target_link_libraries(add_kernel PRIVATE CUDA::cudart)

set_target_properties(add_kernel PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "native"
)