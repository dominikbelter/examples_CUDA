#pragma once

#include <math.h>

__host__ __device__ inline
float objective(float x)
{
    return (x-3) * (x-3) + 9.0f;
}
