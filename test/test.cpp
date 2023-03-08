#include <catch2/catch_test_macros.hpp>
#include <matrix.hpp>

TEST_CASE("Creating object - default constructor")
{
    constexpr std::size_t sg_size = 2;
    mtl::Matrix<int, sg_size, sg_size> m1;
    constexpr std::pair<std::size_t, std::size_t> size{sg_size, sg_size};

    SECTION("Allocation") { REQUIRE(m1.underlying_array() != nullptr); }

    SECTION("Size")
    {
        REQUIRE(m1.size() == size);
        REQUIRE(m1.size_i() == sg_size);
        REQUIRE(m1.size_j() == sg_size);
    }

    SECTION("Reallocation")
    {
        constexpr std::size_t new_sg_size = 3;
        m1.realloc(new_sg_size, new_sg_size);
        std::pair<std::size_t, std::size_t> new_size = {
            new_sg_size,
            new_sg_size};
        REQUIRE(m1.size() == new_size);
        REQUIRE(m1.size_i() == new_sg_size);
        REQUIRE(m1.size_j() == new_sg_size);
    }
}

TEST_CASE("Creating object - fill with value")
{
    constexpr std::size_t sg_size = 2;
    constexpr auto value_to_fill = 3;
    mtl::Matrix<int, sg_size, sg_size> m1(value_to_fill);

    SECTION("Allocation") { REQUIRE(m1.underlying_array() != nullptr); }

    SECTION("Size")
    {
        REQUIRE(
            m1.size() == std::pair<std::size_t, std::size_t>{sg_size, sg_size});
        REQUIRE(m1.size_i() == sg_size);
        REQUIRE(m1.size_j() == sg_size);
    }

    SECTION("Value")
    {
        REQUIRE(m1[0][0] == value_to_fill);
        REQUIRE(m1[0][1] == value_to_fill);
        REQUIRE(m1[1][0] == value_to_fill);
        REQUIRE(m1[1][1] == value_to_fill);
    }
}

TEST_CASE("Creating object - initializer list")
{
    mtl::Matrix<int, 2, 2> m1{1, 2, 3, 4};

    constexpr auto helper = [](){
        mtl::Matrix<int, 2, 2> m1{1, 2, 3, 4, 5};
    };

    SECTION("Allocation") { REQUIRE(m1.underlying_array() != nullptr); }

    SECTION("Invalid initializer_list") {
        REQUIRE_THROWS_AS(helper(), mtl::detail::exceptions::invalid_argument_input);
    }
}

TEST_CASE("Comparison")
{
    SECTION("Equals")
    {
        constexpr auto to_fill = 5;
        mtl::Matrix<int, 4, 4> matrix1(to_fill);
        mtl::Matrix<int, 4, 4> matrix2{
            {5, 5, 5, 5},
            {5, 5, 5, 5},
            {5, 5, 5, 5},
            {5, 5, 5, 5}};

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

    mtl::Matrix<double, 3, 3> matrix2 {1, 2, 3, 4, 5, 6, 7, 8, 9};
    REQUIRE_FALSE(matrix2.is_diagonal());
}

TEST_CASE("Transposition")
{
    mtl::Matrix<int, 2, 2> matrix{1, 2, 3, 4};
    const mtl::Matrix<int, 2, 2> result{1, 3, 2, 4};
    REQUIRE(matrix.transpose() == result);
}

TEST_CASE("Underlying array")
{
    mtl::Matrix<int, 2, 2> matrix { 1, 2, 3, 4 };
    const auto* underlying = matrix.underlying_array();
    REQUIRE(underlying != nullptr);

    for (std::size_t i = 0; i < matrix.size_i(); ++i) {
        for (std::size_t j = 0; j < matrix.size_j(); ++j) {
            REQUIRE(matrix[i][j] == underlying[i][j]);
        }
    }
}

TEST_CASE("Insert") 
{
    constexpr double to_fill = 3.14;
    mtl::Matrix<double, 5, 5> matrix;
    matrix.insert(to_fill);
    for (const auto& elem : matrix) {
        REQUIRE(elem == to_fill);
    }
}

TEST_CASE("Power") {
    mtl::Matrix<int, 2, 2> matrix { 1, 2, 3, 4};
    const mtl::Matrix<int, 2, 2> result { 1, 4, 9, 16};
    REQUIRE(matrix.power(2) == result);
}