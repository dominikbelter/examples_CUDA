#include <iostream>
#include <chrono>

void run_pso_kernel(
    int particles_count,
    int iterations);

void run_pso_modern(
    int particles_count,
    int iterations);

int main()
{
    const int particles = 100000;
    const int iterations = 200;

    {
        auto start =
            std::chrono::high_resolution_clock::now();

        run_pso_kernel(
            particles,
            iterations);

        auto stop =
            std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<
                std::chrono::milliseconds>(
                stop - start);

        std::cout
            << "Kernel version time: "
            << duration.count()
            << " ms\n\n";
    }

    {
        auto start =
            std::chrono::high_resolution_clock::now();

        run_pso_modern(
            particles,
            iterations);

        auto stop =
            std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<
                std::chrono::milliseconds>(
                stop - start);

        std::cout
            << "Modern thrust version time: "
            << duration.count()
            << " ms\n";
    }

    return 0;
}
