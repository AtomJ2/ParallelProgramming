#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <chrono>
#include <omp.h>


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


std::vector<double> matrixVectorMult(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector, int n) {

    std::vector<double> result(n, 0.0);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i] += matrix[i][j] * vector[j];
    return result;
}


std::vector<double> multi_matrixVectorMult(const std::vector<std::vector<double>>& matrix, const std::vector<double>& vector, int numThreads, int n) {

    std::vector<double> result(n, 0.0);

    omp_set_num_threads(numThreads);

#pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            result[i] += matrix[i][j] * vector[j];
    return result;
}


int main() {
    std::string cpuInfo = executeCommand("lscpu");
    std::string serverName = executeCommand("cat /sys/devices/virtual/dmi/id/product_name");
    std::string numaInfo = executeCommand("numactl --hardware");
    std::string osInfo = executeCommand("cat /etc/os-release");

    std::cout << "CPU info:\n" << cpuInfo << std::endl;
    std::cout << "Server: " << serverName << std::endl;
    std::cout << "NUMA-nodes info:\n" << numaInfo << std::endl;
    std::cout << "OS info:\n" << osInfo << std::endl;

    //-------------------------------------------------------------------------------------------------

    std::vector<int> sizes = {20000, 400000};
    std::vector<int> threads = {1, 2, 4, 7, 8, 16, 20, 40};
    std::vector<std::vector<std::pair<long long, double>>> table;
    table.resize(sizes.size(), std::vector<std::pair<long long, double>>(threads.size()));

    long long T = 0.0;
    double S = 0.0;

    for (int i = 0; i < sizes.size(); ++i)
        for (int j = 0; j < sizes.size(); ++j) {
            int n = sizes[i];
            int numThreads = threads[j];

            std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 1.0));
            std::vector<double> vector(n, 2.0);

            auto startTime = std::chrono::high_resolution_clock::now();

            std::vector<double> result;

            result = matrixVectorMult(matrix, vector, n);

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

            T = duration.count();

            startTime = std::chrono::high_resolution_clock::now();

            result = multi_matrixVectorMult(matrix, vector, numThreads, n);

            endTime = std::chrono::high_resolution_clock::now();
            duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

            S = static_cast<double>(T) / static_cast<double>(duration.count());

//            std::cout << "Result: ";
//            for (double val : result) {
//                std::cout << val << " ";
//            }
//            std::cout << std::endl;

            table[i][j] = std::make_pair(T, S);

            std::cout << T << " " << S << std::endl;

        }

    std::cout << "| Size   | Threads | Time (nanoseconds) | Speedup |\n";
    std::cout << "|--------|---------|--------------------|---------|\n";
    for (int i = 0; i < sizes.size(); ++i)
        for (int j = 0; j < threads.size(); ++j) {
            std::cout << "| " << sizes[i] << "    | " << threads[j] << "       | ";
            std::cout << table[i][j].first << "                    | ";
            std::cout << table[i][j].second << "       |\n";
        }

    return 0;
}

