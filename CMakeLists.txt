cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 20)

project(matrix_lib)

add_compile_options(-Wall -Wextra -Wconversion -Wpedantic)

enable_testing()

include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.2.0
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

FetchContent_MakeAvailable(Catch2)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE include/)

add_executable(tests test/test.cpp)
target_link_libraries(tests PRIVATE ${PROJECT_NAME} Catch2::Catch2WithMain)

include(CTest)
include(Catch)
catch_discover_tests(tests)
