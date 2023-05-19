cmake_minimum_required(VERSION 3.18)

project(triton-c-api-test VERSION 0.0.1 LANGUAGES C CXX)

option(TRITON_ENABLE_LOGGING "Include logging support in server" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in server" ON)
option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
set(TRITON_MIN_COMPUTE_CAPABILITY "6.0" CACHE STRING
    "The minimum CUDA compute capability supported by Triton" )

# Ensemble
option(TRITON_ENABLE_ENSEMBLE "Include ensemble support in server" OFF)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()
#
# Dependencies
#
include(FetchContent)

#FetchContent_Declare(
#        repo-common
#        GIT_REPOSITORY https://github.com/triton-inference-server/common.git
#        GIT_TAG ${TRITON_COMMON_REPO_TAG}
#        GIT_SHALLOW ON
#)
FetchContent_Declare(
        repo-core
        GIT_REPOSITORY https://github.com/triton-inference-server/core.git
        GIT_TAG ${TRITON_CORE_REPO_TAG}
        GIT_SHALLOW ON
)
set(TRITON_CORE_HEADERS_ONLY OFF)
#FetchContent_MakeAvailable(repo-common repo-core)
FetchContent_MakeAvailable(repo-core)

#
# CUDA
#
if (${TRITON_ENABLE_GPU})
    find_package(CUDAToolkit REQUIRED)
    find_package(CUDA REQUIRED)
    message(STATUS "Using CUDA ${CUDA_VERSION}")
    set(CUDA_NVCC_FLAGS -std=c++11)

    if (CUDA_VERSION VERSION_GREATER "10.1" OR CUDA_VERSION VERSION_EQUAL "10.1")
        add_definitions(-DTRITON_ENABLE_CUDA_GRAPH=1)
    else ()
        message(WARNING "CUDA ${CUDA_VERSION} does not support CUDA graphs.")
    endif ()
endif () # TRITON_ENABLE_GPU

add_executable(triton-c-api-test main.cpp)

target_include_directories(triton-c-api-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(triton-c-api-test PUBLIC cxx_std_11)

# TRITON_ENABLE_GPU exposed in header so set PUBLIC
if (${TRITON_ENABLE_GPU})
    target_compile_definitions(
            triton-c-api-test
            PUBLIC
            TRITON_ENABLE_GPU=1
            TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
    )
endif () # TRITON_ENABLE_GPU

# TRITON_ENABLE_STATS exposed in header so set PUBLIC
if (${TRITON_ENABLE_STATS})
    target_compile_definitions(
            triton-c-api-test
            PUBLIC TRITON_ENABLE_STATS=1
    )
endif () # TRITON_ENABLE_STATS

target_link_libraries(
        triton-c-api-test
        PUBLIC
        triton-core-serverapi         # from repo-core
        triton-core-serverstub        # from repo-core
)

if (${TRITON_ENABLE_GPU})
    target_link_libraries(
            triton-c-api-test
            PUBLIC
            CUDA::cudart
    )
endif () # TRITON_ENABLE_GPU

#add_library(triton-server SHARED IMPORTED)

#set_target_properties(triton-server
#        PROPERTIES IMPORTED_LOCATION
#        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtritonserver.so)

#target_link_libraries(triton-c-api-test PRIVATE triton-server)
