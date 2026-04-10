#include <algorithm>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>

int main(){
    float k = 0.5;
    float ambient_temp = 20;

    /// CPU only
    std::chrono::steady_clock::time_point beginCPU = std::chrono::steady_clock::now();
    std::vector<float> temp{42, 24, 50};

    auto op = [=](float temp) {
        float diff = ambient_temp - temp;
        return temp + k * diff;
    };

    for (int step = 0; step < 3; step++)
    {
        std::cout << "Step: " << step << ", " << temp[0] << ", " << temp[1] << ", " << temp[2] << "\n";
        std::transform(temp.begin(), temp.end(),
                       temp.begin(), op);
    }
    std::chrono::steady_clock::time_point endCPU = std::chrono::steady_clock::now();
    std::cout << "Time difference for CPU= " <<
        std::chrono::duration_cast<std::chrono::microseconds>(endCPU - beginCPU).count() << "[µs]\n";


    /// and now on GPU and CPU
    std::chrono::steady_clock::time_point beginCUDA = std::chrono::steady_clock::now();
    thrust::universal_vector<float> temp_gpu(3);
    temp_gpu[0] = 42;
    temp_gpu[1] = 24;
    temp_gpu[2] = 50;

    auto op_gpu = [=] __host__ __device__ (float t) {
        float diff = ambient_temp - t;
        return t + k * diff;
    };

    for (int step = 0; step < 3; step++)
    {
        std::cout << "Step: " << step << ", " << temp_gpu[0] << ", " << temp_gpu[1] << ", " << temp_gpu[2] << "\n";
        thrust::transform(thrust::device,
                      temp_gpu.begin(), temp_gpu.end(),
                      temp_gpu.begin(), op_gpu);
    }
    std::chrono::steady_clock::time_point endCUDA = std::chrono::steady_clock::now();
    std::cout << "Time difference for GPU= " <<
        std::chrono::duration_cast<std::chrono::microseconds>(endCUDA - beginCUDA).count() << "[µs]\n";
}
