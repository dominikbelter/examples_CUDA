#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>
#include <limits>
#include <vector>

#include "pso_common.cuh"

#define THREADS 256

constexpr float W  = 0.7f;
constexpr float C1 = 1.4f;
constexpr float C2 = 1.4f;

struct Particle
{
    float x;
    float v;
    float best_x;
    float best_value;
};

__global__ void init_rng(
    curandState* states,
    unsigned long seed)
{
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void init_particles(
    Particle* particles,
    curandState* states,
    int n)
{
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    float x =
        curand_uniform(&states[idx]) * 20.0f - 10.0f;

    particles[idx] =
        {
            x,
            0.0f,
            x,
            objective(x)
        };
}

__global__ void pso_step(
    Particle* particles,
    curandState* states,
    float global_best,
    int n)
{
    const int idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    Particle& p = particles[idx];
    curandState& state = states[idx];

    const float r1 = curand_uniform(&state);
    const float r2 = curand_uniform(&state);

    p.v =
        W  * p.v +
        C1 * r1 * (p.best_x - p.x) +
        C2 * r2 * (global_best - p.x);

    p.x += p.v;

    const float val = objective(p.x);

    if (val < p.best_value)
    {
        p.best_value = val;
        p.best_x = p.x;
    }
}

__global__ void reduce_best(
    Particle* particles,
    Particle* best_particle,
    int n)
{
    __shared__ Particle shared[THREADS];

    const int tid = threadIdx.x;
    const int idx =
        blockIdx.x * blockDim.x + tid;

    Particle p;

    if (idx < n)
        p = particles[idx];
    else
        p.best_value = 1e6;

    shared[tid] = p;

    __syncthreads();

    for (int stride = blockDim.x / 2;
         stride > 0;
         stride >>= 1)
    {
        if (tid < stride)
        {
            if (shared[tid + stride].best_value <
                shared[tid].best_value)
            {
                shared[tid] = shared[tid + stride];
            }
        }

        __syncthreads();
    }

    if (tid == 0)
    {
        best_particle[blockIdx.x] = shared[0];
    }
}

void run_pso_kernel(
    int particles_count,
    int iterations)
{
    Particle* d_particles;
    Particle* d_best_particles;
    curandState* d_states;

    const int blocks =
        (particles_count + THREADS - 1) / THREADS;

    cudaMalloc(
        &d_particles,
        particles_count * sizeof(Particle));

    cudaMalloc(
        &d_states,
        particles_count * sizeof(curandState));

    cudaMalloc(
        &d_best_particles,
        blocks * sizeof(Particle));

    init_rng<<<blocks, THREADS>>>(
        d_states,
        1234);

    init_particles<<<blocks, THREADS>>>(
        d_particles,
        d_states,
        particles_count);

    Particle global_best_particle;
    global_best_particle.best_value =
        std::numeric_limits<float>::max();

    std::vector<Particle> h_best_particles(blocks);

    for (int iter = 0; iter < iterations; ++iter)
    {
        // GPU reduction
        reduce_best<<<blocks, THREADS>>>(
            d_particles,
            d_best_particles,
            particles_count);

        // Copy only block-level minima
        cudaMemcpy(
            h_best_particles.data(),
            d_best_particles,
            blocks * sizeof(Particle),
            cudaMemcpyDeviceToHost);

        // Final CPU reduction
        for (const auto& p : h_best_particles)
        {
            if (p.best_value <
                global_best_particle.best_value)
            {
                global_best_particle = p;
            }
        }

        pso_step<<<blocks, THREADS>>>(
            d_particles,
            d_states,
            global_best_particle.best_x,
            particles_count);
    }

    std::cout << "[Kernel PSO]\n";
    std::cout << "Best x = "
              << global_best_particle.best_x
              << "\n";

    std::cout << "Best value = "
              << global_best_particle.best_value
              << "\n";

    cudaFree(d_particles);
    cudaFree(d_states);
    cudaFree(d_best_particles);
}
