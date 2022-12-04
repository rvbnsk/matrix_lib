#include "../include/matrix.hpp"
#include <catch2/catch_test_macros.hpp>

TEST_CASE("matrix size")
{
    const mtl::Matrix<int, 3, 3> matrix;
    const std::pair<std::size_t, std::size_t> size { 3, 3 };
    REQUIRE(matrix.size() == size);
}

TEST_CASE("is diagonal")
{
    const mtl::Matrix<int, 2, 2> matrix { 1, 0, 0, 1 };
    REQUIRE(matrix.is_diagonal());
}

TEST_CASE("Addition of two matrices")
{
    const mtl::Matrix<float, 2, 2> matrix1 { 1, 1, 1, 1 };
    const mtl::Matrix<float, 2, 2> matrix2 { 1, 1, 1, 1 };
    const mtl::Matrix<float, 2, 2> result { 2, 2, 2, 2};
    REQUIRE(matrix1 + matrix2 == result);
}


