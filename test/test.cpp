#include <catch2/catch_test_macros.hpp>

#include "../include/matrix.hpp"

// TODO: transposition of different sized matrices
// TODO: inequal matrices

TEST_CASE("Comparison")
{
    SECTION("Equals")
    {
        constexpr auto to_fill = 5;
        mtl::Matrix<int, 4, 4> matrix1(to_fill);
        mtl::Matrix<int, 4, 4> matrix2(to_fill);
        REQUIRE(matrix1 == matrix2);
    }
}

TEST_CASE("matrix size")
{
    const mtl::Matrix<int, 3, 3> matrix;
    const std::pair<std::size_t, std::size_t> size{3, 3};
    REQUIRE(matrix.size() == size);
}

TEST_CASE("is diagonal")
{
    const mtl::Matrix<int, 2, 2> matrix{1, 0, 0, 1};
    REQUIRE(matrix.is_diagonal());
}

TEST_CASE("Addition of two matrices")
{
    const mtl::Matrix<float, 2, 2> matrix1{1, 1, 1, 1};
    const mtl::Matrix<float, 2, 2> matrix2{1, 1, 1, 1};
    const mtl::Matrix<float, 2, 2> result{2, 2, 2, 2};
    REQUIRE(matrix1 + matrix2 == result);
}

TEST_CASE("Subtraction of two matrices")
{
    const mtl::Matrix<int, 2, 2> matrix1{2, 2, 2, 2};
    const mtl::Matrix<int, 2, 2> matrix2{3, 3, 3, 3};
    const mtl::Matrix<int, 2, 2> result{-1, -1, -1, -1};
    REQUIRE(matrix1 - matrix2 == result);
}

TEST_CASE("Transposition")
{
    mtl::Matrix<int, 2, 2> matrix{1, 2, 3, 4};
    const mtl::Matrix<int, 2, 2> result{1, 3, 2, 4};
    REQUIRE(matrix.transpose() == result);
}
