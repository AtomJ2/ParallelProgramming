#include <iostream>
#include <omp.h>
#include <new>
#include "laplace2d.h"
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <boost/program_options.hpp>
#include <cuda_runtime.h>
#include "cublas_v2.h"

namespace po = boost::program_options;

int main(int argc, char **argv) {
    int m = 4096;
    int iter_max = 1000;
    double tol = 1.0e-6;

    double error = 1.0;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "Usage: set precision by running -p 123, set grid size by using -n 123 and set maximum number of iterations by -i 123")
        ("precision,p", po::value<double>(&tol)->default_value(1.0e-6), "precision")
        ("grid_size,n", po::value<int>(&m)->default_value(4096), "grid size")
        ("iterations,i", po::value<int>(&iter_max)->default_value(1000), "number of iterations");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    Laplace a(m, m);

    nvtxRangePushA("init");
    a.initialize();
    nvtxRangePop();
    printf("Jacobi relaxation Calculation: %d x %d mesh\n", m, m);

    auto start = std::chrono::high_resolution_clock::now();
    int iter = 0;

    nvtxRangePushA("while");
    while ( error > tol && iter < iter_max )
    {
        nvtxRangePushA("calc");
        a.calcNext();
        nvtxRangePop();
        
        if (iter % 100 == 0){
            nvtxRangePushA("error");
            error = a.calcError();
            nvtxRangePop();
            printf("%5d, %0.6f\n", iter, error);
        }
        
        nvtxRangePushA("swap");
        a.swap();
        nvtxRangePop();

        iter++;
    }
    nvtxRangePop();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration_sec = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << " total: " << duration_sec.count() << "s\n";

    return 0;
}