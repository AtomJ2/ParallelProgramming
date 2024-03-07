#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <numeric>

void matrixVectorMultiplication(const std::vector<std::vector<int>>& matrix, const std::vector<int>& vector, std::vector<int>& result, int start, int end) {
    for (int i = start; i < end; ++i) {
        result[i] = 0;
        for (int j = 0; j < matrix[i].size(); ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

void parallelArrayInit(std::vector<int>& array, int start, int end) {
    for (int i = start; i < end; ++i) {
        array[i] = i;
    }
}

int main() {
    std::vector<int> sizes = {20000, 40000};
    std::vector<int> num_threads = {1, 2, 4, 7, 8, 16, 20, 40};

    std::vector<std::vector<double>> runtimes(num_threads.size(), std::vector<double>(sizes.size()));
    std::vector<std::vector<double>> speedups(num_threads.size(), std::vector<double>(sizes.size()));

    for (int i = 0; i < num_threads.size(); i++) {
        int numThreads = num_threads[i];
        for (int j = 0; j < sizes.size(); j++) {

            int matrixSize = sizes[j];

            std::vector<std::vector<int>> matrix(matrixSize, std::vector<int>(matrixSize, 1));
            std::vector<int> vector(matrixSize, 2);
            std::vector<int> result(matrixSize);


            std::vector<std::thread> threads;
            int chunkSize = matrixSize / numThreads;

            for (int k = 0; k < numThreads; ++k)
                threads.emplace_back(parallelArrayInit, std::ref(vector), k * chunkSize, (k + 1) * chunkSize);

            for (auto& thread : threads)
                thread.join();


            auto start_time = std::chrono::high_resolution_clock::now();

            for (int k = 0; k < numThreads; ++k)
                threads[k] = std::thread(matrixVectorMultiplication, std::ref(matrix), std::ref(vector), std::ref(result), k * chunkSize, (k + 1) * chunkSize);

            for (auto& thread : threads)
                thread.join();

            auto end_time = std::chrono::high_resolution_clock::now();

            double runtime = std::chrono::duration<double>(end_time - start_time).count();

            runtimes[i][j] = runtime;

            double speedup;
            j == 1 ? speedup = runtimes[0][1] / runtime : speedup = runtimes[0][0] / runtime;
            speedups[i][j] = speedup;

            std::cout << "Runtime with " << numThreads << " threads and matrix size " << matrixSize << ": " << runtime << " seconds" << std::endl;
            std::cout << "Speedup with " << numThreads << " threads and matrix size " << matrixSize << ": " << speedup << std::endl << std::endl;
        }
    }

    std::cout << "Summary:" << std::endl;
    for (int i = 0; i < num_threads.size(); i++)
        for (int j = 0; j < sizes.size(); j++)
            std::cout << num_threads[i] << " threads and matrix size " << sizes[j] << ": S = " << speedups[i][j] << ", T = " << runtimes[i][j] << std::endl;

    return 0;
}