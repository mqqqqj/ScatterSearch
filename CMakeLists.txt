cmake_minimum_required(VERSION 3.9)

#指定c和c++编译器
set(CMAKE_C_COMPILER "/home/mqj/bin/gcc-13.2.0/output/bin/gcc")
set(CMAKE_CXX_COMPILER "/home/mqj/bin/gcc-13.2.0/output/bin/g++")

project(annslib)
include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

add_definitions (-std=c++17 -Ofast -lboost -march=native -mavx -mavx2) #  -DINFO

add_subdirectory(src)
add_subdirectory(tests)