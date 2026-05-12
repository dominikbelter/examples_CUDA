#pragma once

#include <math.h>

// do not forget to change the objective in the cu_pso_cpu.cpp
__host__ __device__ inline
float objective(float x)
{
    return (x-3) * (x-3) + 9.0f;
}
