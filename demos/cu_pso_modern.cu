#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
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

struct CompareBest
{
    __host__ __device__
        bool operator()(const Particle& a, const Particle& b) const
    {
        return a.best_value < b.best_value;
    }
};

void run_pso_modern(
    int particles_count,
    int iterations)
{
    thrust::host_vector<Particle> h_particles(particles_count);

    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (auto& p : h_particles)
    {
        p.x = dist(gen);
        p.v = 0.0f;
        p.best_x = p.x;
        p.best_value = objective(p.x);
    }

    thrust::device_vector<Particle> particles = h_particles;

    constexpr float w  = 0.7f;
    constexpr float c1 = 1.4f;
    constexpr float c2 = 1.4f;

    for (int iter = 0; iter < iterations; ++iter)
    {
        // Find global best directly on GPU
        auto best_it = thrust::min_element(
            particles.begin(),
            particles.end(),
            CompareBest());

        const Particle global_best_particle = *best_it;

        const float global_best = global_best_particle.best_x;

        // Update particles
        thrust::for_each(
            particles.begin(),
            particles.end(),
            [=] __device__ (Particle& p)
            {
                thrust::default_random_engine rng(
                    static_cast<unsigned int>(p.x * 1000 + iter));

                thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);

                const float r1 = dist(rng);
                const float r2 = dist(rng);

                p.v =
                    w * p.v +
                    c1 * r1 * (p.best_x - p.x) +
                    c2 * r2 * (global_best - p.x);

                p.x += p.v;

                const float val = objective(p.x);

                if (val < p.best_value)
                {
                    p.best_value = val;
                    p.best_x = p.x;
                }
            });
    }

    // Final global best
    auto best_it = thrust::min_element(
        particles.begin(),
        particles.end(),
        CompareBest());

    const Particle best = *best_it;

    std::cout << "[Modern Thrust PSO]\n";
    std::cout << "Best x = " << best.best_x << "\n";
    std::cout << "Best value = " << best.best_value << "\n";
}
