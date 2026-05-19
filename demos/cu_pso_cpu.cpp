#include <vector>
#include <random>
#include <iostream>
#include <limits>
#include <chrono>
#include <cmath>

struct Particle
{
    float x;
    float v;
    float best_x;
    float best_value;
};

float objective(float x)
{
    return ((x-5) * (x-5) + 1.0f);
}

void run_pso_cpu(int particles_count, int iterations)
{
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::uniform_real_distribution<float> rand01(0.0f, 1.0f);

    std::vector<Particle> particles(particles_count);

    // init particles
    for (auto &p : particles)
    {
        p.x = dist(gen);
        p.v = 0.0f;
        p.best_x = p.x;
        p.best_value = objective(p.x);
    }

    float global_best = particles[0].best_x;
    float global_best_val = particles[0].best_value;

    // find initial global best
    for (const auto &p : particles)
    {
        if (p.best_value < global_best_val)
        {
            global_best_val = p.best_value;
            global_best = p.best_x;
        }
    }

    // PSO loop
    for (int iter = 0; iter < iterations; ++iter)
    {
        for (auto &p : particles)
        {
            float r1 = rand01(gen);
            float r2 = rand01(gen);

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
        }

        // update global best
        for (const auto &p : particles)
        {
            if (p.best_value < global_best_val)
            {
                global_best_val = p.best_value;
                global_best = p.best_x;
            }
        }
    }

    std::cout << "[CPU PSO]\n";
    std::cout << "Best x = " << global_best << "\n";
    std::cout << "Best value = " << global_best_val << "\n";
}
