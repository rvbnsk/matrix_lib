set(NAME tests)

add_executable(${NAME})

target_sources(${NAME} PRIVATE src/test.cpp)

include(FetchContent)

FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.12.0
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

FetchContent_MakeAvailable(Catch2)

target_link_libraries(tests PRIVATE
        Catch2::Catch2WithMain
        Matrix
)

enable_testing()

include(CTest)
include(Catch)
catch_discover_tests(tests)