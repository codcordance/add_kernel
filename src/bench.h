#pragma once

#include "data.h"

template <data::FloatFormat T>
using LauncherFn = float (*)(const data::DeviceData<T>&, cudaStream_t);

template <data::FloatFormat T>
void bench(const char* name, std::size_t size, int repeats, cudaStream_t s, LauncherFn<T> launch);

void measurePCIeBandwidth();