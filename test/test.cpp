#include <catch2/catch_test_macros.hpp>
#include <matrix.hpp>

TEST_CASE("Creating object - default constructor")
{
    constexpr std::size_t sg_size = 2;
    mtl::Matrix<int, sg_size, sg_size> m1;
    constexpr std::pair<std::size_t, std::size_t> size{ sg_size, sg_size };

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
        std::pair<std::size_t, std::size_t> new_size = { new_sg_size,
                                                         new_sg_size };
        REQUIRE(m1.size() == new_size);
        REQUIRE(m1.size_i() == new_sg_size);
        REQUIRE(m1.size_j() == new_sg_size);
    }
}

TEST_CASE("Creating object - big matrix size")
{
    constexpr std::size_t size = 10000;
    mtl::Matrix<int, size, size> m1;

    SECTION("Allocation") { REQUIRE(m1.underlying_array() != nullptr); }
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
            m1.size()
            == std::pair<std::size_t, std::size_t>{ sg_size, sg_size });
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
    constexpr auto initialize_with_valid_list = []() {
        return mtl::Matrix<int, 2, 2>{ 1, 2, 3, 4 };
    };

    REQUIRE_NOTHROW(initialize_with_valid_list());

    const auto m1 = initialize_with_valid_list();

    SECTION("Allocation") { REQUIRE(m1.underlying_array() != nullptr); }

    constexpr auto initialize_with_invalid_list = []() {
        return mtl::Matrix<int, 2, 2>{ 1, 2, 3, 4, 5 };
    };

    SECTION("Invalid initializer_list")
    {
        REQUIRE_THROWS_AS(
            initialize_with_invalid_list(),
            mtl::detail::exceptions::invalid_argument_input);
    }
}

TEST_CASE("Creating object - nested initializer list")
{
    constexpr auto initialize_with_valid_list = []() {
        return mtl::Matrix<int, 3, 3>{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
    };

    REQUIRE_NOTHROW(initialize_with_valid_list());

    const auto m1 = initialize_with_valid_list();

    SECTION("Allocation") { REQUIRE(m1.underlying_array() != nullptr); }

    constexpr auto initialize_with_invalid_list = []() {
        return mtl::Matrix<int, 3, 3>{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8 } };
    };

    SECTION("Invalid initializer_list")
    {
        REQUIRE_THROWS_AS(
            initialize_with_invalid_list(),
            mtl::detail::exceptions::invalid_argument_input);
    }
}

TEST_CASE("Operator= with std::initializer_list")
{
    SECTION("Copy initializer_list")
    {
        mtl::Matrix<int, 2, 2> m1;
        const std::initializer_list<int> list{ 1, 2, 3, 4 };

        m1 = list;
        REQUIRE(m1[0][0] == 1);
        REQUIRE(m1[0][1] == 2);
        REQUIRE(m1[1][0] == 3);
        REQUIRE(m1[1][1] == 4);
    }

    SECTION("Move initializer_list")
    {
        mtl::Matrix<int, 2, 2> m1;

        m1 = { 1, 2, 3, 4 };
        REQUIRE(m1[0][0] == 1);
        REQUIRE(m1[0][1] == 2);
        REQUIRE(m1[1][0] == 3);
        REQUIRE(m1[1][1] == 4);
    }
}

TEST_CASE("Comparison")
{
    SECTION("Equals")
    {
        constexpr auto to_fill = 5;
        mtl::Matrix<int, 4, 4> matrix1(to_fill);
        mtl::Matrix<int, 4, 4> matrix2{ { 5, 5, 5, 5 },
                                        { 5, 5, 5, 5 },
                                        { 5, 5, 5, 5 },
                                        { 5, 5, 5, 5 } };

        REQUIRE(matrix1 == matrix2);
    }

    SECTION("Equals with std::initializer_list")
    {
        constexpr auto to_fill = 2;
        mtl::Matrix<int, 3, 3> matrix1(to_fill);
        const auto initializer_list = { 2, 2, 2, 2, 2, 2, 2, 2, 2 };

        REQUIRE(matrix1 == initializer_list);

        mtl::Matrix<int, 2, 2> matrix2{ 1, 2, 3, 4 };
        std::initializer_list<int> list = { 1, 2, 3, 4 };
        REQUIRE(matrix2 == list);
    }
}

TEST_CASE("matrix size")
{
    const mtl::Matrix<int, 3, 3> matrix;
    const std::pair<std::size_t, std::size_t> size{ 3, 3 };
    REQUIRE(matrix.size() == size);
}

TEST_CASE("operator[]")
{
    SECTION("Row")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
        std::vector<int> row{ 1, 2 };
        REQUIRE(matrix[0].get_row() == row);
    }

    SECTION("Assign row")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
        std::vector<int> new_row{ 0, 0 };
        matrix[0] = new_row;

        REQUIRE(matrix[0].get_row() == new_row);
        REQUIRE(matrix.underlying_array()[0][1] == new_row.at(1));

        std::initializer_list<int> row{ 0, 0, 3, 4 };
        REQUIRE(matrix == row);
    }

    SECTION("Crow")
    {
        const mtl::Matrix<int, 2, 2> matrix{ 4, 5, 6, 7 };
        const std::vector<int> row{ 6, 7 };
        REQUIRE(matrix[1].get_row() == row);
    }
}

TEST_CASE("operator[][]")
{
    SECTION("Matrix")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        int value = 1;

        for (std::size_t row = 0; row < matrix.size_i(); ++row) {
            for (std::size_t col = 0; col < matrix.size_j(); ++col) {
                REQUIRE(matrix[row][col] == value++);
            }
        }
    }

    SECTION("const Matrix")
    {
        const mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        int value = 1;

        for (std::size_t row = 0; row < matrix.size_i(); ++row) {
            for (std::size_t col = 0; col < matrix.size_j(); ++col) {
                REQUIRE(matrix[row][col] == value++);
            }
        }
    }
}

TEST_CASE("Range based for loop")
{
    SECTION("Matrix")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        int value = 1;

        for (auto elem : matrix) { REQUIRE(elem == value++); }
    }

    SECTION("const Matrix")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        int value = 1;

        for (const auto elem : matrix) { REQUIRE(elem == value++); }
    }
}

TEST_CASE("is diagonal")
{
    const mtl::Matrix<int, 2, 2> matrix{ 1, 0, 0, 1 };
    REQUIRE(matrix.is_diagonal());

    mtl::Matrix<double, 3, 3> matrix2{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    REQUIRE_FALSE(matrix2.is_diagonal());
}

TEST_CASE("Transposition")
{
    SECTION("TC01")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
        const mtl::Matrix<int, 2, 2> result{ 1, 3, 2, 4 };
        REQUIRE(matrix.transpose() == result);
    }

    SECTION("TC02")
    {
        mtl::Matrix<double, 3, 2> matrix{ { 1, 2 }, { 3, 4 }, { 5, 6 } };
        const mtl::Matrix<double, 2, 3> result{ { 1, 3, 5 }, { 2, 4, 6 } };
        REQUIRE(matrix.transpose() == result);
    }
}

TEST_CASE("Underlying array")
{
    mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
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
    for (const auto& elem : matrix) { REQUIRE(elem == to_fill); }
}

TEST_CASE("Power")
{
    mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
    const mtl::Matrix<int, 2, 2> result{ 7, 10, 15, 22 };
    REQUIRE(matrix.power(2) == result);
}

TEST_CASE("Multiplication - quadratic matrix")
{
    const mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
    const auto result_values = { 7, 10, 15, 22 };

    auto result = matrix * matrix;

    REQUIRE(result.size() == matrix.size());
    REQUIRE(result == result_values);
}

TEST_CASE("Multiplication - non quadratic matrix")
{
    const mtl::Matrix<int, 3, 2> matrix1{ 1, 2, 3, 4, 5, 6 };
    const mtl::Matrix<int, 2, 3> matrix2{ 1, 2, 3, 4, 5, 6 };
    const auto result_values = { 9, 12, 15, 19, 26, 33, 29, 40, 51 };
    const auto result = matrix1 * matrix2;

    REQUIRE(result.size() == std::pair<std::size_t, std::size_t>{ 3, 3 });
    REQUIRE(result == result_values);
}

TEST_CASE("Addition")
{
    const mtl::Matrix<double, 2, 2> matrix{ 1, 2, 3, 4 };
    const mtl::Matrix<double, 2, 2> result{ 2, 4, 6, 8 };
    REQUIRE(matrix + matrix == result);
}

TEST_CASE("Adjection")
{
    const mtl::Matrix<double, 2, 2> matrix{ 1, 2, 3, 4 };
    const mtl::Matrix<double, 2, 2> result{ 0, 0, 0, 0 };
    REQUIRE(matrix - matrix == result);
}

TEST_CASE("Multiplication by scalar")
{
    const mtl::Matrix<double, 2, 2> matrix{ 1, 2, 3, 4 };
    const int scalar = 5;
    const mtl::Matrix<double, 2, 2> result{ 5, 10, 15, 20 };
    REQUIRE(matrix * scalar == result);
}

TEST_CASE("Multiplication by vector") { const std::vector<int> vec{ 0, 1, 2 }; }