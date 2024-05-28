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


// разница
__global__ void subtr_arr(const double *A, const double *Anew, double *subtr_res , int m) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if ((i > 1) && (i < m) && (j > 1) && (j < m))
        subtr_res[OFFSET(i, j, m)] = fabs(A[OFFSET(i, j, m)] - Anew[OFFSET(i, j, m)]);
}


int main(int argc, char **argv) {
    int m = 128;
    int iter_max = 1000000;
    double tol = 1.0e-6;
    double err = 1.0;

    po::options_description desc("Allowed options");
    desc.add_options()
            ("help", "Produce help message")
            ("precision", po::value<double>(&tol)->default_value(tol), "Set precision")
            ("grid-size", po::value<int>(&m)->default_value(m), "Set grid size")
            ("iterations", po::value<int>(&iter_max)->default_value(iter_max), "Set number of iterations");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    std::unique_ptr<double[]> A_ptr(new double[m*m]);
    std::unique_ptr<double[]> Anew_ptr(new double[m*m]);
    std::unique_ptr<double[]> subtr_temp_ptr(new double[m * m]);
    double *temp;

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

    double *d_error_ptr = d_unique_ptr_error.get();
    cudaErr = cudaMalloc((void**)&d_error_ptr, sizeof(double));

    void *d_temp_storage = d_unique_ptr_temp_storage.get();
    size_t temp_storage_bytes = 0;

    // вызываем DeviceReduce чтобы посмотреть сколько требуется памяти для временного хранения
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, Anew, d_error_ptr, m*m, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    printf("Jacobi relaxation Calculation: %d x %d mesh\n", m, m);

    #pragma acc data copyin(A[:m*m], Anew[:m*m], subtr_temp[:m*m])
    {
        int iter = 0;
        nvtxRangePushA("while");
        auto start_time = std::chrono::high_resolution_clock::now();
        while (err > tol && iter < iter_max) {
            nvtxRangePushA("calcNext");
            #pragma acc parallel loop async(1) independent collapse(2) vector vector_length(m) gang num_gangs(m) present(A,Anew)
            for (int i = 1; i < m - 1; i++)
                for (int j = 1; j < m - 1; j++)
                    Anew[OFFSET(i, j, m)] = 0.25 * (A[OFFSET(i, j + 1, m)] + A[OFFSET(i, j - 1, m)] + A[OFFSET(i - 1, j, m)] + A[OFFSET(i + 1, j, m)]);
            nvtxRangePop();
            if (iter % 1000 == 0) {
                nvtxRangePushA("calcError");
                err = 0;
                #pragma acc host_data use_device(A, Anew, subtr_temp)
                {
                    subtr_arr<<<grid, block, 0, stream>>>(A, Anew, subtr_temp, m);
                    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, subtr_temp, d_error_ptr, m * m, stream);
                    // асинхронное копирование из d_error_ptr в err между девайсом и хостом 
                    cudaErr = cudaMemcpyAsync(&err, d_error_ptr, sizeof(double), cudaMemcpyDeviceToHost, stream);
                }
                nvtxRangePop();
            }

            nvtxRangePushA("swap");
            temp = A;
            A = Anew;
            Anew = temp;
            nvtxRangePop();

            if (iter % 1000 == 0)
                printf("%5d, %0.6f\n", iter, err);

            iter++;
        }
        nvtxRangePop();
        printf("%5d, %0.6f\n", iter, err);
        #pragma acc update host(A[:m*m], Anew[:m*m])

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> runtime = end_time - start_time;
        printf("total: %f s\n", runtime.count());
    }

    // std::ofstream output("output.txt");
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < m; j++)
    //         output << std::left << std::setw(10) << A[OFFSET(i, j, m)] << " ";
    //     output << std::endl;
    // }

    return 0;
}