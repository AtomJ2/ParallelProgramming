#include <iostream>
#include <boost/program_options.hpp>
#include <omp.h>
#include <new>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <fstream>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_store.cuh>


#define OFFSET(x, y, m) (((x) * (m)) + (y))


namespace po = boost::program_options;


// указатель для управления памятью на устройстве
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T,std::function<void(T*)>>;


// выделение памяти на устройстве
template<typename T>
T* cuda_new(size_t size) {
    T *d_ptr;
    cudaMalloc((void **)&d_ptr, sizeof(T) * size);
    return d_ptr;
}


// освобождение ресурсов
template<typename T>
void cuda_free(T *dev_ptr) {
    cudaFree(dev_ptr);
}


__global__ void subtr_arr(const double *A, const double *Anew, double *subtr_res , int m) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if ((i >= 0) && (i < m) && (j >= 0) && (j < m))
        subtr_res[OFFSET(i, j, m)] = fabs(A[OFFSET(i, j, m)] - Anew[OFFSET(i, j, m)]);
}


__global__ void calc_mean(double *A, double *Anew, int m, bool flag) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (flag) {
        if ((i > 0) && (i < m - 1) && (j > 0) && (j < m - 1))
            A[OFFSET(j, i, m)] = 0.25 * (Anew[OFFSET(j, i + 1, m)] + Anew[OFFSET(j, i - 1, m)] + Anew[OFFSET(j - 1, i, m)] + Anew[OFFSET(j + 1, i, m)]);
    }
    else {
        if ((i > 0) && (i < m - 1) && (j > 0) && (j < m - 1))
            Anew[OFFSET(j, i, m)] = 0.25 * (A[OFFSET(j, i + 1, m)] + A[OFFSET(j, i - 1, m)] + A[OFFSET(j - 1, i, m)] + A[OFFSET(j + 1, i, m)]);
    }
}


int main(int argc, char **argv) {
    int m = 1024;
    int iter_max = 1000000;
    double precision = 1.0e-6;
    double err = 1.0;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "Produce help message")
            ("precision", po::value<double>(&precision)->default_value(precision), "Set precision")
            ("grid-size", po::value<int>(&m)->default_value(m), "Set grid size")
            ("iterations", po::value<int>(&iter_max)->default_value(iter_max), "Set number of iterations");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::unique_ptr<double[]> A_ptr(new double[m*m]);
    std::unique_ptr<double[]> Anew_ptr(new double[m*m]);
    std::unique_ptr<double[]> subtr_temp_ptr(new double[m * m]);

    double* A = A_ptr.get();
    double* Anew = Anew_ptr.get();
    double* subtr_temp = subtr_temp_ptr.get();

    nvtxRangePushA("init");
    double corners[] = {10.0, 20.0, 30.0, 20.0};
    memset(A, 0, m * m * sizeof(double));
    A[0] = corners[0];
    A[m - 1] = corners[1];
    A[m * m - 1] = corners[2];
    A[m * (m - 1)] = corners[3];

    double number_of_steps = m - 1;
    double top_side_step = (double)abs(corners[0] - corners[1]) / number_of_steps;
    double right_side_step = (double)abs(corners[1] - corners[2]) / number_of_steps;
    double bottom_side_step = (double)abs(corners[2] - corners[3]) / number_of_steps;
    double left_side_step = (double)abs(corners[3] - corners[0]) / number_of_steps;

    double top_side_min = std::min(corners[0], corners[1]);
    double right_side_min = std::min(corners[1], corners[2]);
    double bottom_side_min = std::min(corners[2], corners[3]);
    double left_side_min = std::min(corners[3], corners[0]);

    for (int i = 1; i < m - 1; i++) {
        A[i] = top_side_min + i * top_side_step;
        A[m * i] = left_side_min + i * left_side_step;
        A[(m - 1) + m * i] = right_side_min + i * right_side_step;
        A[m * (m - 1) + i] = bottom_side_min + i * bottom_side_step;
    }
    std::memcpy(Anew, A, m * m * sizeof(double));
    nvtxRangePop();

    dim3 grid(64, 64);
    dim3 block(16, 16);

    cudaError_t cudaErr = cudaSuccess;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cuda_unique_ptr<double> d_unique_ptr_error(cuda_new<double>(0), cuda_free<double>);
    cuda_unique_ptr<void> d_unique_ptr_temp_storage(cuda_new<void>(0), cuda_free<void>);

    cuda_unique_ptr<double> d_unique_ptr_A(cuda_new<double>(0), cuda_free<double>);
    cuda_unique_ptr<double> d_unique_ptr_Anew(cuda_new<double>(0), cuda_free<double>);
    cuda_unique_ptr<double> d_unique_ptr_subtr_temp(cuda_new<double>(0), cuda_free<double>);

    // выделение памяти и перенос на устройство
    double *d_error_ptr = d_unique_ptr_error.get();
    cudaErr = cudaMalloc((void**)&d_error_ptr, sizeof(double));

    double *d_A = d_unique_ptr_A.get();
    cudaErr = cudaMalloc((void **)&d_A, m * m * sizeof(double));

    double *d_Anew = d_unique_ptr_Anew.get();
    cudaErr = cudaMalloc((void **)&d_Anew, m * m * sizeof(double));

    double *d_subtr_temp = d_unique_ptr_subtr_temp.get();
    cudaErr = cudaMalloc((void **)&d_subtr_temp, m * m * sizeof(double));

    cudaErr = cudaMemcpy(d_A, A, m * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaErr = cudaMemcpy(d_Anew, Anew, m * m * sizeof(double), cudaMemcpyHostToDevice);

    // проверка памяти для редукции
    void *d_temp_storage = d_unique_ptr_temp_storage.get();
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_Anew, d_error_ptr, m * m, stream);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", m, m);

    bool graph_created = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    int iter = 0;

    nvtxRangePushA("while");
    auto start_time = std::chrono::high_resolution_clock::now();
    while (err > precision && iter < iter_max) {
        if(!graph_created) {
            nvtxRangePushA("createGraph");
            // начало захвата операций на потоке stream
            cudaErr = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for (int i = 0; i < 100; i++)
                calc_mean<<<grid, block, 0, stream>>>(d_A, d_Anew, m, (i % 2 == 1));
            // завершение захвата операций
            cudaErr = cudaStreamEndCapture(stream, &graph);
            nvtxRangePop();
            // создаем исполняемый граф
            cudaErr = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graph_created = true;
        }
        // старт графа
        nvtxRangePushA("startGraph");
        // запускаем исполняемый граф
        cudaErr = cudaGraphLaunch(instance, stream);
        nvtxRangePop();
        iter += 100;
        if (iter % 100 == 0){
            nvtxRangePushA("calcError");
            subtr_arr<<<grid, block, 0, stream>>>(d_A, d_Anew, d_subtr_temp, m);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_subtr_temp, d_error_ptr, m * m, stream);
            cudaErr = cudaMemcpy(&err, d_error_ptr, sizeof(double), cudaMemcpyDeviceToHost);
            nvtxRangePop();
        }
        if (iter % 1000 == 0)
            printf("%5d, %0.6f\n", iter, err);
    }
    nvtxRangePop();
    printf("%5d, %0.6f\n", iter, err);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> runtime = end_time - start_time;

    printf("total: %f s\n", runtime.count());
    cudaErr = cudaMemcpy(A, d_A, m * m * sizeof(double), cudaMemcpyDeviceToHost);

    // std::ofstream out("out.txt");
    // for (int i = 0; i < m; i++){
    //     for (int j = 0; j < m; j++)
    //         out << std::left << std::setw(10) << A[OFFSET(i, j, m)] << " ";
    //     out << std::endl;
    // }

    return 0;
}
