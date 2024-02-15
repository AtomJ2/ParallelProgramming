cmake_minimum_required(VERSION 3.22)
project(parallel_task2)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

add_executable(parallel_task2 test2.cpp)

target_link_libraries(parallel_task2 PRIVATE OpenMP::OpenMP_CXX)