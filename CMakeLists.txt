cmake_minimum_required(VERSION 3.24)
project(moving_on_with_nvidia_gpus
        LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR}/bin)

find_package(CUDAToolkit REQUIRED)
enable_testing()

add_subdirectory(common)
add_subdirectory(chapters/ch01_architecture)
add_subdirectory(chapters/ch02_environments)
add_subdirectory(chapters/ch03_advanced_cuda)
add_subdirectory(chapters/ch04_profiling)