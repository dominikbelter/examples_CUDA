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

// Kernel function to add the elements of two arrays
__global__ void add(int n, float *x, float *y, float *out)
{
    for (int i = 0; i < n; i++)
        out[i] = x[i] + y[i];
}

int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out;

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    std::chrono::steady_clock::time_point beginCPU = std::chrono::steady_clock::now();
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

    // Initialize array
    for(int i = 0; i < N; i++){
        d_a[i] = 1.0f; d_b[i] = 2.0f;
    }

    std::chrono::steady_clock::time_point beginCUDA = std::chrono::steady_clock::now();
    /// add vectors using one thread
    // add<<<1, 1>>>(N, d_a, d_b, d_out);

    // std::cout << "Executing the same code on the GPU is significantly slower than on the CPU.\n";
    // std::cout << "The key point here is the kernel launch add<<<1, 1>>>. The second parameter \n";
    // std::cout << "specifies the number of threads. In this case, the kernel is executed using a single\n";
    // std::cout << "thread. GPU cores are slower than CPU cores, and there is also data transfer\n";
    // std::cout << "overhead, which is why the GPU execution is much slower in this case. In the next\n";
    // std::cout << "example, we will increase the number of threads to 256.\n";
    /// increase number of threads to 256 - comment the line above and uncomment the line below
    // add<<<1, 256>>>(N, d_a, d_b, d_out);

    /// the most flexible way:
    std::cout << "CUDA-enabled GPUs consist of many processors grouped into so-called \n";
    std::cout << "Streaming Multiprocessors (SMs). Each of them supports multiple threads.\n";
    std::cout << "To fully utilize the capabilities of the GPU, it is necessary to divide \n";
    std::cout << "the task into blocks and threads within those blocks: \n";
    std::cout << "<<<number_of_blocks, number_of_threads>>> (or <<<number_of_blocks, block_size>>>).\n";
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout << "numBlocks " << numBlocks << "\n";
    std::cout << "blockSize " << blockSize << "\n";
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
