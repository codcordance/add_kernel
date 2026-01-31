#include <iostream>

void add_thread_block_run() {
    constexpr int N = 1<<20;
    std::cout << "N: " << N << std::endl;

    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f; y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    add_thread_block_kernel<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    }
    std::cout << "Max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
}

#include <cuda_bf16.h> // Nécessaire pour __nv_bfloat16

// Kernel modifié pour BF16
__global__ void add_thread_block_kernel_bf16(int n, __nv_bfloat16 *x, __nv_bfloat16 *y) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        // L'addition directe n'est pas toujours définie selon l'architecture,
        // on utilise souvent les intrinsèques pour être sûr
        y[i] = __hadd(x[i], y[i]);
    }
}

void add_thread_block_run_bf16() {
    constexpr int N = 1<<20;

    __nv_bfloat16 *x, *y;
    // La taille en mémoire est maintenant divisée par 2 (2 octets par élément)
    cudaMallocManaged(&x, N * sizeof(__nv_bfloat16));
    cudaMallocManaged(&y, N * sizeof(__nv_bfloat16));

    for (int i = 0; i < N; i++) {
        // Conversion explicite de float vers bfloat16 pour l'initialisation CPU
        x[i] = __float2bfloat16(1.0f);
        y[i] = __float2bfloat16(2.0f);
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    add_thread_block_kernel_bf16<<<numBlocks, blockSize>>>(N, x, y);
    cudaDeviceSynchronize();

    // Vérification (re-conversion en float pour le calcul d'erreur)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        float valY = __bfloat162float(y[i]);
        maxError = fmax(maxError, fabs(valY - 3.0f));
    }
    std::cout << "Max error (BF16): " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
}

DataArrays init_arrays(const int n)
{
    float *x, *y;
    cudaMallocManaged(&x, n*sizeof(float));
    cudaMallocManaged(&y, n*sizeof(float));

    for (int i = 0; i < n; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    return {x, y};
}

float process_arrays(const int n, float *x, float *y) {

}

int main() {
    constexpr int n = 1<<20;
    std::cout << "Will use arrays of size n = " << n << std::endl;

    float *x, *y;
    cudaMallocManaged(&x, n*sizeof(float));
    cudaMallocManaged(&y, n*sizeof(float));

    std::cout << "Running  NAIVE" << std::endl;
    add_naive_main();

    std::cout << "ADD BLOCK" << std::endl;
    add_block_main();

    std::cout << "ADD THREAD BLOCK" << std::endl;
    add_thread_block_run();

    std::cout << "ADD THREAD BLOCK BF16" << std::endl;
    add_thread_block_run_bf16();

    return 0;
}