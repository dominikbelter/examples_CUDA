#include <cuda_runtime.h>
#include "cuda_conv.cuh"

__global__ void conv_kernel(unsigned char* input,
                            char* mask,
                            unsigned char* output,
                            int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = rows * cols;

    if (idx >= size) return;

    int r = idx / cols;
    int c = idx % cols;

    int sum = 0;

    for(int dr = -1; dr <= 1; dr++){
        for(int dc = -1; dc <= 1; dc++){
            int rr = r + dr;
            int cc = c + dc;

            if(rr >= 0 && rr < rows && cc >= 0 && cc < cols){
                int mask_idx = (dr+1)*3 + (dc+1);
                sum += input[rr * cols + cc] * mask[mask_idx];
            }
        }
    }

    if (sum < 0) sum = 0;
    else if (sum > 255) sum = 255;

    output[idx] = static_cast<unsigned char>(sum);
}

void computeConvCUDA(unsigned char* input,
                     char* mask,
                     unsigned char* output,
                     int rows,
                     int cols)
{
    size_t size = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    size_t bytes = size * sizeof(unsigned char);

    unsigned char *d_input = nullptr;
    unsigned char *d_output = nullptr;
    char *d_mask = nullptr;

    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_mask, 9 * sizeof(char));

    cudaMemcpy(d_input, input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, mask, 9 * sizeof(char), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = static_cast<int>((size + threadsPerBlock - 1) / threadsPerBlock);

    conv_kernel<<<blocks, threadsPerBlock>>>(d_input, d_mask, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_mask);
}
