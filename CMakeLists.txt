cmake_minimum_required(VERSION 3.27.7)

set(CMAKE_CXX_STANDARD 20)

project(MatrixLib LANGUAGES CXX)

add_subdirectory(matrix_lib)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  add_subdirectory(test)
endif()