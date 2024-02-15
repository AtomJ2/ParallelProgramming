#include <iostream>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <vector>


double f(double x) {
    return std::sin(x);
}


double integrate_omp(int num_threads, int nsteps) {
    double result = 0.0;
    double step = 1.0 / nsteps;

#pragma omp parallel for num_threads(num_threads) reduction(+:result)
    for (int i = 0; i < nsteps; i++) {
        double x = (i + 0.5) * step;
        result += f(x);
    }

    result *= step;

    return result;
}


int main() {
    int nsteps = 40000000;
    std::vector<int> num_threads = {1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<double> runtimes(num_threads.size());
    std::vector<double> speedups(num_threads.size());

    double sequential_runtime;
    double sequential_result;

    {
        auto start_time = std::chrono::high_resolution_clock::now();
        sequential_result = integrate_omp(1, nsteps);
        auto end_time = std::chrono::high_resolution_clock::now();
        sequential_runtime = std::chrono::duration<double>(end_time - start_time).count();
    }

    std::cout << "Sequential result: " << sequential_result << std::endl;
    std::cout << "Sequential runtime: " << sequential_runtime << " seconds" << std::endl;

    for (int i = 0; i < num_threads.size(); i++) {
        int num_thread = num_threads[i];

        omp_set_num_threads(num_thread);

        auto start_time = std::chrono::high_resolution_clock::now();
        double parallel_result = integrate_omp(num_thread, nsteps);
        auto end_time = std::chrono::high_resolution_clock::now();
        double parallel_runtime = std::chrono::duration<double>(end_time - start_time).count();

        runtimes[i] = parallel_runtime;
        speedups[i] = sequential_runtime / parallel_runtime;

        std::cout << "Runtime with " << num_thread << " threads: " << parallel_runtime << " sec" << std::endl;
        std::cout << "Speedup with " << num_thread << " threads: " << speedups[i] << std::endl << std::endl;
    }

    std::cout << "Summary:" << std::endl;
    for (int i = 0; i < num_threads.size(); i++)
        std::cout << num_threads[i] << " threads: S = " << speedups[i] << ", T = " << runtimes[i] << std::endl;

    return 0;
}