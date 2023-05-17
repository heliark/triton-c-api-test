cmake_minimum_required(VERSION 3.18)

project(triton-c-api-test VERSION 0.0.1 LANGUAGES C CXX)

add_executable(triton-c-api-test main.cpp)

target_include_directories(triton-c-api-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_features(triton-c-api-test PUBLIC cxx_std_11)

add_library(triton-server SHARED IMPORTED)

set_target_properties(triton-server
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtritonserver.so)

target_link_libraries(triton-c-api-test PRIVATE triton-server)
