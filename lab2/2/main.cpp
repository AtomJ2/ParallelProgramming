#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>


// 4.0 / (1.0 + x^2) [0;1]
double integrate_omp(int nsteps) {
    double sum = 0.0;
    double step = 1.0 / nsteps;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < nsteps; i++) {
        double x = (i + 0.5) * step;
        double fx = 4.0 / (1.0 + x * x);
        sum += fx;
    }

    return sum * step;
}


int main() {
    int nsteps = 40000000;
    
    std::vector<int> num_threads = {1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<double> speedup(num_threads.size());
    std::vector<double> runtimes(num_threads.size());

    double serial_time, parallel_time;

    auto start_time = std::chrono::high_resolution_clock::now();
    double result_serial = integrate_omp(nsteps);
    auto end_time = std::chrono::high_resolution_clock::now();
    serial_time = std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "Time: " << serial_time << " sec" << std::endl << std::endl;

    for (int i = 0; i < num_threads.size(); i++) {
        int num_thread = num_threads[i];

        omp_set_num_threads(num_thread);

        start_time = std::chrono::high_resolution_clock::now();
        double result_parallel = integrate_omp(nsteps);
        end_time = std::chrono::high_resolution_clock::now();
        parallel_time = std::chrono::duration<double>(end_time - start_time).count();

        runtimes[i] = parallel_time;
        double speed = serial_time / parallel_time;
        speedup[i] = speed;

        std::cout << "Time with " << num_thread << " threads: " << parallel_time << " seconds" << std::endl;
        std::cout << "S with " << num_thread << " threads: " << speed << std::endl << std::endl;
    }

    std::cout << "\nSummary:\n";
    for (int i = 0; i < num_threads.size(); i++)
        std::cout << num_threads[i] << " threads: S = " << speedup[i] << ", T = " << runtimes[i] << std::endl;

    return 0;
}