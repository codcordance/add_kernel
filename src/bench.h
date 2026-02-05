#pragma once

#include "data.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <type_traits>
#include <utility>

void printBench(const char* name, float timeMsMean, double bandwidthGbs, long double errCum);

template <data::FloatFormat T>
using LauncherFn = float (*)(const data::DeviceData<T>&, cudaStream_t);

template <data::FloatFormat T, class Launch>
requires std::invocable<Launch&, const data::DeviceData<T>&, cudaStream_t> &&
         std::convertible_to<std::invoke_result_t<Launch&, const data::DeviceData<T>&, cudaStream_t>, float>
void bench(const char* name,
                  const std::size_t size,
                  const int repeats,
                  cudaStream_t s,
                  Launch&& launch) {
    float timeMsCum = 0.0f;
    long double errCum = 0.0L;

    // 1 warmup iteration
    for (int i = 0; i < repeats + 1; ++i) {
        data::HostData<T> h(size);
        data::DeviceData<T> d(size);
        h.init(12345);

        data::copyH2DAsync(h, d, s);
        CUDA_CHECK(cudaStreamSynchronize(s));

        const auto timeMs = static_cast<float>(std::forward<Launch>(launch)(d, s));
        CUDA_CHECK(cudaStreamSynchronize(s));

        data::copyD2HAsync(d, h, s);
        CUDA_CHECK(cudaStreamSynchronize(s));

        if (i > 0) {
            timeMsCum += timeMs;
            errCum += static_cast<long double>(h.error());
        }
    }

    const float timeMsMean = timeMsCum / static_cast<float>(repeats);

    const std::uint64_t bytes = static_cast<std::uint64_t>(size) * 3ull * sizeof(T); // 3 = 2 read + 1 write
    const double bandwidthGbs = (static_cast<double>(bytes) / 1.0e6) / static_cast<double>(timeMsMean);

    printBench(name, timeMsMean, bandwidthGbs, errCum);
}

//FP32 overload
template <class Launch>
requires std::invocable<Launch&, const data::DeviceData<float>&, cudaStream_t> &&
         std::convertible_to<std::invoke_result_t<Launch&, const data::DeviceData<float>&, cudaStream_t>, float>
void bench(const char* name,
                  const std::size_t size,
                  const int repeats,
                  cudaStream_t s,
                  Launch&& launch) {
    bench<float>(name, size, repeats, s, std::forward<Launch>(launch));
}

void measurePCIeBandwidth();
