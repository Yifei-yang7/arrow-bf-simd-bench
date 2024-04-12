cmake_minimum_required(VERSION 3.15)
project(arrow_bf_simd_bench)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}/cmake")
set(DEPS_PREFIX "_deps")
include(External_arrow)
include(External_nanobench)
include(External_fmt)

add_executable(arrow_bf_simd_bench
        bench/main.cpp)
target_link_libraries(arrow_bf_simd_bench arrow_static)
target_link_libraries(arrow_bf_simd_bench nanobench::nanobench)
target_link_libraries(arrow_bf_simd_bench fmt::fmt)