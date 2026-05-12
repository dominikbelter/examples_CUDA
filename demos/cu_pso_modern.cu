#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/extrema.h>
#include <thrust/random.h>

#include <iostream>
#include <random>
#include <limits>
#include "pso_common.cuh"

struct Particle
{
    float x;
    float v;
    float best_x;
    float best_value;
};

void run_pso_modern(
    int particles_count,
    int iterations)
{
    thrust::host_vector<Particle> h_particles(particles_count);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (int i = 0; i < particles_count; ++i)
    {
        float x = dist(gen);

        h_particles[i].x = x;
        h_particles[i].v = 0.0f;
        h_particles[i].best_x = x;
        h_particles[i].best_value = objective(x);
    }

    thrust::device_vector<Particle> particles = h_particles;

    float global_best = 0.0f;
    float global_best_val = std::numeric_limits<float>::max();

    for (int iter = 0; iter < iterations; ++iter)
    {
        thrust::host_vector<Particle> temp = particles;

        for (auto& p : temp)
        {
            if (p.best_value < global_best_val)
            {
                global_best_val = p.best_value;
                global_best = p.best_x;
            }
        }

        thrust::for_each(
            particles.begin(),
            particles.end(),
            [=] __device__ (Particle& p)
            {
                thrust::random::default_random_engine rng;
                rng.discard((unsigned int)(p.x * 1000));

                thrust::random::uniform_real_distribution<float> dist(0.0f, 1.0f);

                float r1 = dist(rng);
                float r2 = dist(rng);

                const float w = 0.7f;
                const float c1 = 1.4f;
                const float c2 = 1.4f;

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
            });
    }

    std::cout << "[Modern Thrust PSO]\n";
    std::cout << "Best x = " << global_best << "\n";
    std::cout << "Best value = " << global_best_val << "\n";
}
