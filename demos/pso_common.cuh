#pragma once

#include <math.h>

// do not forget to change the objective in the cu_pso_cpu.cpp
__host__ __device__ inline
float objective(float x)
{
    return (x-5) * (x-5) + 1.0f;
}
