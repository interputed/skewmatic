cmake_minimum_required(VERSION 2.8)
project(skewmatic CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build)
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

set(CXX_OPTIMIZATION_FLAGS "-std=c++11 -O2")
set(CXX_ERROR_FLAGS "-Wall -Werror -Wextra -pedantic")
set(CMAKE_CXX_FLAGS "${CXX_OPTIMIZATION_FLAGS} ${CXX_ERROR_FLAGS}")

add_subdirectory(src)
