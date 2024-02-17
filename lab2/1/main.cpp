#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>

/*
std::string executeCommand(const std::string& command) {
    std::string result = "";
    char buffer[128];
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe) {
        while (!feof(pipe))
            if (fgets(buffer, 128, pipe) != nullptr)
                result += buffer;
        pclose(pipe);
    }
    return result;
}
*/

void init(std::vector<std::vector<double>>& matrix, std::vector<double>& vector, int matrix_size) {
#pragma omp parallel for
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++)
            matrix[i][j] = i + j;
        vector[i] = i;
    }
}

void multiplication(int num_threads, std::vector<std::vector<double>> matrix, std::vector<double> vector, int threads) {
#pragma omp parallel num_threads(threads)
    {
#pragma omp for schedule(guided)
        for (int i = 0; i < matrix.size(); i++) {
            for (int j = 0; j < matrix.size(); j++) {
                matrix[i][j] *= vector[j];
            }
        }
    }
}

int main() {
    /*
    std::string cpuInfo = executeCommand("lscpu");
    std::string serverName = executeCommand("cat /sys/devices/virtual/dmi/id/product_name");
    std::string numaInfo = executeCommand("numactl --hardware");
    std::string osInfo = executeCommand("cat /etc/os-release");

    std::cout << "CPU info:\n" << cpuInfo << std::endl;
    std::cout << "Server: " << serverName << std::endl;
    std::cout << "NUMA-nodes info:\n" << numaInfo << std::endl;
    std::cout << "OS info:\n" << osInfo << std::endl;
    */

    //-------------------------------------------------------------------------------------------------

    std::vector<int> num_threads = {1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<int> matrix_sizes = {20000, 40000};
    std::vector<std::vector<double>> runtimes(num_threads.size(), std::vector<double>(matrix_sizes.size()));
    std::vector<std::vector<double>> speedups(num_threads.size(), std::vector<double>(matrix_sizes.size()));

    for (int i = 0; i < num_threads.size(); i++) {
        for (int j = 0; j < matrix_sizes.size(); j++) {
            int threads = num_threads[i];
            int matrix_size = matrix_sizes[j];

            // omp_set_num_threads(threads);

            std::vector<std::vector<double>> matrix(matrix_size, std::vector<double>(matrix_size));
            std::vector<double> vector(matrix_size);

            init(matrix, vector, matrix_size);

            auto start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(threads)
            {
#pragma omp for schedule(guided)
                for (int q = 0; q < matrix_size; q++)
                    for (int w = 0; w < matrix_size; w++)
                        matrix[q][w] *= vector[w];
            }

//            multiplication(threads, matrix, vector, threads);

            auto end_time = std::chrono::high_resolution_clock::now();
            double runtime = std::chrono::duration<double>(end_time - start_time).count();

            runtimes[i][j] = runtime;

            double speedup;
            j == 1 ? speedup = runtimes[0][1] / runtime : speedup = runtimes[0][0] / runtime;
            speedups[i][j] = speedup;

            std::cout << "Runtime with " << threads << " threads and matrix size " << matrix_size << ": " << runtime << " seconds" << std::endl;
            std::cout << "Speedup with " << threads << " threads and matrix size " << matrix_size << ": " << speedup << std::endl << std::endl;
        }
    }

    std::cout << "Summary:" << std::endl;
    for (int i = 0; i < num_threads.size(); i++)
        for (int j = 0; j < matrix_sizes.size(); j++)
            std::cout << num_threads[i] << " threads and matrix size " << matrix_sizes[j] << ": S = " << speedups[i][j] << ", T = " << runtimes[i][j] << std::endl;

    return 0;
}
