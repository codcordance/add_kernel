#include "../kernels/naive_old.cuh"
#include "../cuda_utils.h"

__global__ void addNaiveKernel(const int n, const float* x, float* y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveRestrictKernel(const int n, const float* __restrict__ x, float* __restrict__ y) {
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveUnroll4NoTailRestrictKernel(const int n,
                                                    const float* __restrict__ x,
                                                    float* __restrict__ y) {
    for (int i = 0; i < n; i += 4) {
        y[i]     = x[i]     + y[i];
        y[i + 1] = x[i + 1] + y[i + 1];
        y[i + 2] = x[i + 2] + y[i + 2];
        y[i + 3] = x[i + 3] + y[i + 3];
    }
}


__global__ void addNaiveSizeTKernel(const size_t n, const float* x, float* y) {
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveSizeTRestrictKernel(const size_t n, const float* __restrict__ x, float* __restrict__ y) {
    for (size_t i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

__global__ void addNaiveUnroll2RestrictKernel(const int n, const float* __restrict__ x, float* __restrict__ y) {
    const int n2 = n >> 1;

    const auto* __restrict__ x2 = reinterpret_cast<const float2*>(x);
    auto* __restrict__ y2 = reinterpret_cast<float2*>(y);

    for (int i = 0; i < n2; ++i) {
        float2 a = x2[i];
        float2 b = y2[i];
        b.x += a.x;
        b.y += a.y;
        y2[i] = b;
    }

    // tail
    if (n & 1) {
        const int i = n - 1;
        y[i] = x[i] + y[i];
    }
}

__global__ void addNaiveUnroll2MultRestrictKernel(const int n2, const float2* __restrict__ x2, float2* __restrict__ y2) {
    for (int i = 0; i < n2; ++i) {
        float2 a = x2[i];
        float2 b = y2[i];
        b.x += a.x;
        b.y += a.y;
        y2[i] = b;
    }
}

__global__ void addNaiveFloat4RestrictKernel(const int n, const float* __restrict__ x, float* __restrict__ y) {
    const int n4 = n >> 2;

    const auto* __restrict__ x4 = reinterpret_cast<const float4*>(x);
    auto* __restrict__ y4 = reinterpret_cast<float4*>(y);

    for (int i = 0; i < n4; ++i) {
        float4 a = x4[i];
        float4 b = y4[i];
        b.x += a.x;
        b.y += a.y;
        b.z += a.z;
        b.w += a.w;
        y4[i] = b;
    }

    for (int i = (n4 << 2); i < n; ++i) {
        y[i] = x[i] + y[i];
    }
}

__global__ void addNaiveUnroll4MultKernel(const int n4, const float4* x4, float4* y4) {
    for (int i = 0; i < n4; ++i) {
        float4 a = x4[i];
        float4 b = y4[i];
        b.x += a.x;
        b.y += a.y;
        b.z += a.z;
        b.w += a.w;
        y4[i] = b;
    }
}

__global__ void addNaiveUnroll4MultRestrictKernel(const int n4, const float4* __restrict__ x4, float4* __restrict__ y4) {
    for (int i = 0; i < n4; ++i) {
        float4 a = x4[i];
        float4 b = y4[i];
        b.x += a.x;
        b.y += a.y;
        b.z += a.z;
        b.w += a.w;
        y4[i] = b;
    }
}


float addNaive(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveKernel<<<1, 1, 0, s>>>(static_cast<int>(d.n), d.dx, d.dy);
    });
}

float addNaiveRestrict(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveRestrictKernel<<<1, 1, 0, s>>>(static_cast<int>(d.n), d.dx, d.dy);
    });
}

float addNaiveUnroll4NoTailRestrict(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveUnroll4NoTailRestrictKernel<<<1, 1, 0, s>>>(static_cast<int>(d.n), d.dx, d.dy);
    });
}

float addNaiveSizeT(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveSizeTKernel<<<1, 1, 0, s>>>(d.n, d.dx, d.dy);
    });
}

float addNaiveSizeTRestrict(const data::DeviceData<float> &d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveSizeTRestrictKernel<<<1, 1, 0, s>>>(d.n, d.dx, d.dy);
    });
}

float addNaiveUnroll2Restrict(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveUnroll2RestrictKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n), d.dx, d.dy
        );
    });
}

float addNaiveUnroll2MultRestrict(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveUnroll2MultRestrictKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n) >> 1, reinterpret_cast<const float2*>(d.dx), reinterpret_cast<float2*>(d.dy)
        );
    });
}

float addNaiveUnroll4Restrict(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveFloat4RestrictKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n), d.dx, d.dy
        );
    });
}

float addNaiveUnroll4Mult(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveUnroll4MultKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n) >> 2, reinterpret_cast<const float4*>(d.dx), reinterpret_cast<float4*>(d.dy)
        );
    });
}

float addNaiveFloat4NoTailRestrict(const data::DeviceData<float>& d, cudaStream_t s) {
    return timeKernel(s, [&] {
        addNaiveUnroll4MultRestrictKernel<<<1, 1, 0, s>>>(
            static_cast<int>(d.n) >> 2, reinterpret_cast<const float4*>(d.dx), reinterpret_cast<float4*>(d.dy)
        );
    });
}

