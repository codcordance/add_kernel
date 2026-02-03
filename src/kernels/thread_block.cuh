#pragma once

#include "../data.h"

float addThreadBlock(const data::DeviceData<float> &d, cudaStream_t s);
float addThreadBlockRestrict(const data::DeviceData<float> &d, cudaStream_t s);