#include <iostream>
#include <chrono>

void run_pso_kernel(int particles_count, int iterations);
void run_pso_modern(int particles_count, int iterations);
void run_pso_cpu(int particles_count, int iterations);

template<typename Func>
void benchmark(
    const std::string& name,
    Func func,
    int particles,
    int iterations)
{
    const auto start =
        std::chrono::high_resolution_clock::now();

    func(particles, iterations);

    const auto stop =
        std::chrono::high_resolution_clock::now();

    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            stop - start);

    std::cout
        << name
        << ": "
        << duration.count()
        << " ms\n";
}

int main()
{
    constexpr int particles  = 100000;
    constexpr int iterations = 200;

    benchmark(
        "Kernel version time",
        run_pso_kernel,
        particles,
        iterations);

    benchmark(
        "Modern thrust version time",
        run_pso_modern,
        particles,
        iterations);

    benchmark(
        "CPU PSO time",
        run_pso_cpu,
        particles,
        iterations);

    return 0;
}
