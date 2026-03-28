#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

// Kernel function to initialize three arrays
__global__
    void init(int n, float *x, float *y, float *out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        x[i] = 1.0;
        y[i] = 2.0;
        out[i] = 0.0;
    }
}

// Kernel function to add the elements of two arrays
__global__
    void add(int n, float *x, float *y, float *out)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride){
        out[i] = x[i] + y[i];
    }
}

int main(){
    std::cout << "In the previous examples, we used the cudaMallocManaged() \n";
    std::cout << "function to allocate memory for variables processed by the GPU. \n";
    std::cout << "At the same time, we used the CPU to initialize these variables. \n";
    std::cout << "In this case, we used so-called unified memory, which is accessible \n";
    std::cout << "to both the CPU and the GPU. The GPU drivers automatically decide \n";
    std::cout << "where to physically place the data.\n\n";

    std::cout << "It is also possible to use the cudaMemPrefetchAsync() function, \n";
    std::cout << "which prefetches memory to the GPU (not all GPUs support this). \n";
    std::cout << "In such a case, initializing variables on the CPU may not be possible.\n\n";

    std::cout << "In the next example, we will also measure the time required to \n";
    std::cout << "initialize the data. For the GPU case, the variables will be \n";
    std::cout << "initialized directly on the GPU. Therefore, a kernel for initializing\n";
    std::cout << "these variables on the GPU has been added.\n\n";

    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    std::chrono::steady_clock::time_point beginCPU = std::chrono::steady_clock::now();
    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // add vectors
    vector_add(out, a, b, N);
    std::chrono::steady_clock::time_point endCPU = std::chrono::steady_clock::now();
    std::cout << "Time difference for CPU= " <<
        std::chrono::duration_cast<std::chrono::microseconds>(endCPU - beginCPU).count() << "[µs]\n";

    // print result
    for(int i = 0; i < 10; i++){
        std::cout << out[i] << "\n";
    }

    // GPU CUDA
    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&d_a, N*sizeof(float));
    cudaMallocManaged(&d_b, N*sizeof(float));
    cudaMallocManaged(&d_out, N*sizeof(float));

    // Initialize and add arrays
    std::chrono::steady_clock::time_point beginCUDA = std::chrono::steady_clock::now();
    // add vectors
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    init<<<numBlocks, blockSize>>>(N, d_a, d_b, d_out);
    add<<<numBlocks, blockSize>>>(N, d_a, d_b, d_out);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point endCUDA = std::chrono::steady_clock::now();
    std::cout << "Time difference for GPU= " <<
        std::chrono::duration_cast<std::chrono::microseconds>(endCUDA - beginCUDA).count() << "[µs]\n";

    // print result
    for(int i = 0; i < 10; i++){
        std::cout << d_out[i] << "\n";
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
    free(a);
    free(b);
    free(out);

    std::cout << "Finished\n";
}
