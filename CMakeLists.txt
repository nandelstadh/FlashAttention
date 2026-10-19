cmake_minimum_required(VERSION 3.24)
project(flash_attention LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20) # libtorch 2.14 requires C++20
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared) # Same shared cudart as libtorch, not a second static copy
# libtorch takes the architecture from TORCH_CUDA_ARCH_LIST (slurm/common.sh) and warns
# that it ignores this variable; the warning is harmless
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89) # L40s
endif()

find_package(Torch REQUIRED) # Pass -DCMAKE_PREFIX_PATH=<libtorch dir>

add_executable(main main.cu flash_attention.cu)
target_link_libraries(main "${TORCH_LIBRARIES}")
