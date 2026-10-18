cmake_minimum_required(VERSION 3.24)
project(flash_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90) # H100
endif()

find_package(Torch REQUIRED) # Pass -DCMAKE_PREFIX_PATH=<libtorch dir>

add_executable(main main.cu flash_attention.cu)
target_link_libraries(main "${TORCH_LIBRARIES}")
