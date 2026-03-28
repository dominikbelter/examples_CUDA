#include <iostream>

int main()
{
    try {
        std::cout << "Important: before running this example, check your CUDA driver \n";
        std::cout << "and NVCC installation. Type in the terminal:\n";
        std::cout << "$ nvidia-smi\n";
        std::cout << "$ lsb_release -a\n";
        std::cout << "$ nvcc --version\n";
        std::cout << "CUDA examples. Run demo programs:\n";
        std::cout << "demo_CUDA1 - check GPU info\n";
        std::cout << "demo_CUDA_add - add some numbers on GPU\n";
        std::cout << "demo_CUDA_add_kernel - add two vectors on GPU\n";
        std::cout << "demo_CUDA_initialization - add two vectors on GPU and initialize data on GPU\n";
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
    return 0;
}
