#pragma once

#include <cuda_runtime.h>

void computeConvCUDA(unsigned char* input,
                     char* mask,
                     unsigned char* output,
                     int rows,
                     int cols);
