#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <limits>
#include <vector>
#include "pso_common.cuh"

#define THREADS 256

struct Particle
{
    float x;
    float v;
    float best_x;
    float best_value;
};

__global__ void init_rng(curandState* states, unsigned long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void init_particles(
    Particle* particles,
    curandState* states,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    float x = curand_uniform(&states[idx]) * 20.0f - 10.0f;

    particles[idx].x = x;
    particles[idx].v = 0.0f;
    particles[idx].best_x = x;
    particles[idx].best_value = objective(x);
}

__global__ void pso_step(
    Particle* particles,
    curandState* states,
    float global_best,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    const float w = 0.7f;
    const float c1 = 1.4f;
    const float c2 = 1.4f;

    float r1 = curand_uniform(&states[idx]);
    float r2 = curand_uniform(&states[idx]);

    Particle& p = particles[idx];

    p.v =
        w * p.v +
        c1 * r1 * (p.best_x - p.x) +
        c2 * r2 * (global_best - p.x);

    p.x += p.v;

    float val = objective(p.x);

    if (val < p.best_value)
    {
        p.best_value = val;
        p.best_x = p.x;
    }
}

void run_pso_kernel(
    int particles_count,
    int iterations)
{
    Particle* d_particles;
    curandState* d_states;

    cudaMalloc(&d_particles, particles_count * sizeof(Particle));
    cudaMalloc(&d_states, particles_count * sizeof(curandState));

    int blocks = (particles_count + THREADS - 1) / THREADS;

    init_rng<<<blocks, THREADS>>>(d_states, 1234);
    init_particles<<<blocks, THREADS>>>(
        d_particles,
        d_states,
        particles_count);

    std::vector<Particle> h_particles(particles_count);

    float global_best = 0.0f;
    float global_best_val = std::numeric_limits<float>::max();

    for (int iter = 0; iter < iterations; ++iter)
    {
        cudaMemcpy(
            h_particles.data(),
            d_particles,
            particles_count * sizeof(Particle),
            cudaMemcpyDeviceToHost);

        for (auto& p : h_particles)
        {
            if (p.best_value < global_best_val)
            {
                global_best_val = p.best_value;
                global_best = p.best_x;
            }
        }

        pso_step<<<blocks, THREADS>>>(
            d_particles,
            d_states,
            global_best,
            particles_count);
    }

    std::cout << "[Kernel PSO]\n";
    std::cout << "Best x = " << global_best << "\n";
    std::cout << "Best value = " << global_best_val << "\n";

    cudaFree(d_particles);
    cudaFree(d_states);
}
