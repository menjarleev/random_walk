# set(CMAKE_THREAD_LIBS_INIT "-lpthread")
# set(CMAKE_HAVE_THREADS_LIBRARY 1)
# set(CMAKE_USE_WIN32_THREADS_INIT 0)
# set(CMAKE_USE_PTHREADS_INIT 1)
# set(THREADS_PREFER_PTHREAD_FLAG ON)
cmake_minimum_required(VERSION 3.5)
project(random_walk LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

add_library(cuda_lib STATIC
            ./rwalk.cu)

# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_80,code=sm_80;--std=c++11)

# cuda_compile(cuda_objs ${CUDA_SRC})

add_executable(${PROJECT_NAME} rwalk_kernel.cc)

set_property(TARGET ${PROJECT_NAME}
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
        PUBLIC
            cuda_lib
            ${LIBS})

