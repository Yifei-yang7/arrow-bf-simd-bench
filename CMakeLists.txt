cmake_minimum_required(VERSION 3.15)
project(arrow_bf_simd_bench)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}/cmake")
set(DEPS_PREFIX "_deps")
include(External_arrow)
include(External_nanobench)
include(External_fmt)

add_executable(bf_bench
        bench/bf_bench.cpp)
target_link_libraries(bf_bench arrow_static)
target_link_libraries(bf_bench nanobench::nanobench)
target_link_libraries(bf_bench fmt::fmt)

add_library(hash
        hash/hash.cpp hash/hash.hpp)

add_executable(hash_bench
        bench/hash_bench.cpp)
target_link_libraries(hash_bench hash)
target_link_libraries(hash_bench nanobench::nanobench)
target_link_libraries(hash_bench fmt::fmt)
target_include_directories(hash_bench PUBLIC hash)