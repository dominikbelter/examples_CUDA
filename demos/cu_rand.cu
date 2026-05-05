/*
 * This program uses the host CURAND API to generate pseudorandom floats.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <chrono>
#include <random>
#include <iostream>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

    int main(int argc, char *argv[])
    {
        size_t n = 1e7;
        curandGenerator_t gen;
        float *devData, *hostData;

        std::chrono::steady_clock::time_point beginGPU = std::chrono::steady_clock::now();
        /* Allocate n floats on host */
        hostData = (float *)calloc(n, sizeof(float));

        /* Allocate n floats on device */
        CUDA_CALL(cudaMalloc((void **)&devData, n*sizeof(float)));

        /* Create pseudo-random number generator */
        CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

        /* Set seed */
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(NULL)));

        /* Generate n floats on device */
        //CURAND_CALL(curandGenerateUniform(gen, devData, n));
        CURAND_CALL(curandGenerateNormal(gen, devData, n, 0.0, 1.0));

        /* Copy device memory to host */
        CUDA_CALL(cudaMemcpy(hostData, devData, n * sizeof(float), cudaMemcpyDeviceToHost));

        std::chrono::steady_clock::time_point endGPU = std::chrono::steady_clock::now();
        std::cout << "Time difference GPU = " << std::chrono::duration_cast<std::chrono::microseconds>(endGPU - beginGPU).count() << "[µs]" << std::endl;

        /* Show result */
        for(int i = 0; i < 10; i++) {
            printf("%1.4f ", hostData[i]);
        }
        printf("\n");

        /* Cleanup */
        CURAND_CALL(curandDestroyGenerator(gen));
        CUDA_CALL(cudaFree(devData));
        free(hostData);

        // CPU
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,1.0);

        std::vector<float> vectCPU(n, 0.0);
        for (int i=0; i<n; ++i) {
            vectCPU[i] = distribution(generator);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference CPU = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

        return EXIT_SUCCESS;
    }
