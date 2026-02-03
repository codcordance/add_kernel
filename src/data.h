#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <optional>
#include <random>

#include "cuda_utils.h"

namespace data {
    template <typename T>
    concept FloatFormat =
        std::is_same_v<T, float> ||
        std::is_same_v<T, __nv_bfloat16>;

    template <FloatFormat T>
    T to_T(float v) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) return __float2bfloat16(v);
        return static_cast<T>(v);
    }

    template <FloatFormat T>
    float from_T(T v) {
        if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(v);
        return static_cast<float>(v);
    }


    template <FloatFormat T>
    struct HostData {
        std::size_t n = 0;
        T* hx = nullptr;
        T* hy = nullptr;
        float* sum = nullptr;

        HostData() = default;
        explicit HostData(const std::size_t n_) { allocate(n_); }

        HostData(const HostData&) = delete;
        HostData& operator=(const HostData&) = delete;

        HostData(HostData&& o) noexcept
        : n(o.n), hx(o.hx), hy(o.hy), sum(o.sum) {
            o.n = 0;
            o.hx = nullptr;
            o.hy = nullptr;
            o.sum = nullptr;
        }

        HostData& operator=(HostData&& o) noexcept {
            if (this != &o) {
                release();
                n = o.n;
                hx = o.hx;
                hy = o.hy;
                sum = o.sum;
                o.n = 0;
                o.hx = nullptr;
                o.hy = nullptr;
                o.sum = nullptr;
            }
            return *this;
        }

        ~HostData() { release(); }

        void allocate(const std::size_t n_) {
            release();
            n = n_;
            CUDA_CHECK(cudaMallocHost(&hx, n * sizeof(T)));
            CUDA_CHECK(cudaMallocHost(&hy, n * sizeof(T)));
            CUDA_CHECK(cudaMallocHost(&sum, n * sizeof(float)));
        }

        void release() noexcept {
            if (hx) cudaFreeHost(hx);
            if (hy) cudaFreeHost(hy);
            if (sum) cudaFreeHost(sum);
            n = 0;
            hx = nullptr;
            hy = nullptr;
            sum = nullptr;
        }

        void init(const std::optional<std::uint32_t> seed = std::nullopt, const float min = 0.0f, const float max = 1.0f) {
            std::mt19937 gen(seed ? *seed : std::random_device{}());
            std::uniform_real_distribution dis(min, max);

            for (std::size_t i = 0; i < n; i++) {
                const float x = dis(gen);
                const float y = dis(gen);
                hx[i] = to_T<T>(x);
                hy[i] = to_T<T>(y);
                sum[i] = x + y;
            }
        }

        [[nodiscard]] long double error() const {
            long double errCum = 0L;
            for (std::size_t i = 0; i < n; i++) {
                const float err = from_T<T>(hy[i]) - sum[i];
                errCum += std::fabsf(err);
            }
            return errCum;
        }
    };

    template <FloatFormat T>
    struct DeviceData {
        std::size_t n = 0;
        T* dx = nullptr;
        T* dy = nullptr;

        DeviceData() = default;
        explicit DeviceData(const std::size_t n_) { allocate(n_); }

        DeviceData(const DeviceData&) = delete;
        DeviceData& operator=(const DeviceData&) = delete;

        DeviceData(DeviceData&& o) noexcept : n(o.n), dx(o.dx), dy(o.dy) {
            o.n = 0;
            o.dx = nullptr;
            o.dy = nullptr;
        }

        DeviceData& operator=(DeviceData&& o) noexcept {
            if (this != &o) {
                release();
                n = o.n;
                dx = o.dx;
                dy = o.dy;
                o.n = 0;
                o.dx = nullptr;
                o.dy = nullptr;
            }
            return *this;
        }

        ~DeviceData() { release(); }

        void allocate(const std::size_t n_) {
            release();
            n = n_;
            CUDA_CHECK(cudaMalloc(&dx, n * sizeof(T)));
            CUDA_CHECK(cudaMalloc(&dy, n * sizeof(T)));
        }

        void release() noexcept {
            if (dx) cudaFree(dx);
            if (dy) cudaFree(dy);
            n = 0;
            dx = nullptr;
            dy = nullptr;
        }
    };

    template <FloatFormat T>
    void copyH2DAsync(const HostData<T>& h, const DeviceData<T>& d, cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(d.dx, h.hx, h.n * sizeof(T), cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(d.dy, h.hy, h.n * sizeof(T), cudaMemcpyHostToDevice, s));
    }

    template <FloatFormat T>
    void copyD2HAsync(const DeviceData<T>& d, const HostData<T>& h, cudaStream_t s) {
        CUDA_CHECK(cudaMemcpyAsync(h.hy, d.dy, h.n * sizeof(T), cudaMemcpyDeviceToHost, s));
    }

} // namespace data
