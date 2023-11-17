#include <catch2/catch_test_macros.hpp>
#include <matrix.hpp>
#include <numeric>

TEST_CASE("Creating object - default constructor")
{
    constexpr std::size_t sg_size = 2;
    mtl::Matrix<int, sg_size, sg_size> m1;
    constexpr std::pair<std::size_t, std::size_t> size{ sg_size, sg_size };

    SECTION("Allocation")
    {
        REQUIRE(m1.underlying_array() != nullptr);
        REQUIRE_FALSE(m1.is_reallocated());
    }

    SECTION("Size")
    {
        REQUIRE(m1.size() == size);
        REQUIRE(m1.row_size() == sg_size);
        REQUIRE(m1.col_size() == sg_size);
    }

    SECTION("Reallocation")
    {
        constexpr std::size_t new_sg_size = 3;
        m1.realloc(new_sg_size, new_sg_size);
        std::pair<std::size_t, std::size_t> new_size = { new_sg_size,
                                                         new_sg_size };
        REQUIRE(m1.size() == new_size);
        REQUIRE(m1.row_size() == new_sg_size);
        REQUIRE(m1.col_size() == new_sg_size);
    }
}

TEST_CASE("Underlying array")
{
    mtl::Matrix<int, 2, 2> matrix{};
    const auto* underlying = matrix.underlying_array();

    REQUIRE(underlying != nullptr);

    matrix = { 1, 2, 3, 4 };

    REQUIRE(underlying != nullptr);

    for (std::size_t i = 0; i < matrix.row_size(); ++i) {
        for (std::size_t j = 0; j < matrix.col_size(); ++j) {
            REQUIRE(matrix[i][j] == underlying[i][j]);
        }
    }
}

TEST_CASE("Reallocation")
{
    constexpr std::size_t base_size = 2;
    mtl::Matrix<int, base_size, base_size> m{ 1, 2, 3, 4 };

    REQUIRE_FALSE(m.is_reallocated());

    REQUIRE(m.row_size() == base_size);
    REQUIRE(m.col_size() == base_size);

    constexpr std::size_t new_size = 3;
    m.realloc(new_size, new_size);

    REQUIRE(m.is_reallocated());

    REQUIRE(m.row_size() == new_size);
    REQUIRE(m.col_size() == new_size);
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
        REQUIRE(m1.row_size() == sg_size);
        REQUIRE(m1.col_size() == sg_size);
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
            std::invalid_argument);
    }
}

TEST_CASE("Deduction guide")
{
    mtl::Matrix m1(5);
    const std::pair<std::size_t, std::size_t> expected_size{ 1, 1 };
    REQUIRE(m1.size() == expected_size);
    REQUIRE(m1.at(0, 0) == 5);
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
            std::invalid_argument);
    }
}

TEST_CASE("Copying matrix")
{
    SECTION("Same type CTOR")
    {
        const mtl::Matrix<int, 2, 2> m1{ 1, 2, 3, 4 };
        const mtl::Matrix<int, 2, 2> m2{ m1 };
        REQUIRE(m1 == m2);
    }

    SECTION("Different type CTOR")
    {
        const mtl::Matrix<int, 2, 2> m1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> m2{ m1 };
        REQUIRE(m1 == m2);
    }

    SECTION("Same type assignment operator")
    {
        const mtl::Matrix<int, 2, 2> m1{ 1, 2, 3, 4 };
        mtl::Matrix<int, 2, 2> m2;
        m2 = m1;
        REQUIRE(m1 == m2);
    }

    SECTION("Different type assignment operator")
    {
        const mtl::Matrix<int, 2, 2> m1{ 1, 2, 3, 4 };
        mtl::Matrix<double, 2, 2> m2;
        m2 = m1;
        REQUIRE(m1 == m2);
    }
}

TEST_CASE("Moving matrix")
{
    SECTION("Move Constructor Moves Values Correctly")
    {
        mtl::Matrix<int, 2, 3> matrix1{ 1, 2, 3, 4, 5, 6 };
        mtl::Matrix<int, 2, 3> matrix2 = std::move(matrix1);
        REQUIRE(matrix1.underlying_array() == nullptr);
        REQUIRE(matrix2.underlying_array() != nullptr);

        constexpr std::pair<std::size_t, std::size_t> size_prev{ 0, 0 };
        constexpr std::pair<std::size_t, std::size_t> size_val{ 2, 3 };
        REQUIRE(matrix1.size() == size_prev);
        REQUIRE(matrix2.size() == size_val);

        int value{ 0 };
        for (std::size_t i = 0; i < matrix2.row_size(); ++i) {
            for (std::size_t j = 0; j < matrix2.col_size(); ++j) {
                REQUIRE(matrix2[i][j] == ++value);
            }
        }
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

    SECTION("Not Equals Different Sizes")
    {
        constexpr auto to_fill = 5;
        mtl::Matrix<int, 4, 4> matrix1(to_fill);
        mtl::Matrix<int, 3, 3> matrix2(to_fill);

        REQUIRE_FALSE(matrix1 == matrix2);
        REQUIRE(matrix1 != matrix2);
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

        for (std::size_t row = 0; row < matrix.row_size(); ++row) {
            for (std::size_t col = 0; col < matrix.col_size(); ++col) {
                REQUIRE(matrix[row][col] == value++);
            }
        }
    }

    SECTION("const Matrix")
    {
        const mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        int value = 1;

        for (std::size_t row = 0; row < matrix.row_size(); ++row) {
            for (std::size_t col = 0; col < matrix.col_size(); ++col) {
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

    SECTION("Matrix with ref")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        constexpr int new_value = 4;
        for (auto& elem : matrix) { elem = new_value; }

        const std::initializer_list<int> expected_result = { 4, 4, 4, 4 };
        REQUIRE(matrix == expected_result);
    }

    SECTION("With reallocation")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        matrix.realloc(3, 3);
        REQUIRE(matrix.is_reallocated());
        constexpr int value = 1;
        for (const auto elem : matrix) { REQUIRE(elem == value); }
    }

    SECTION("const Matrix")
    {
        const mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };

        int value = 1;

        for (const auto& elem : matrix) { REQUIRE(elem == value++); }
    }

    SECTION("const Matrix with copied elements")
    {
        const mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
        int value = 1;
        for (auto elem : matrix) {
            elem = 1;
            REQUIRE(elem);
        }
        for (const auto& elem : matrix) { REQUIRE(elem == value++); }
    }
}

TEST_CASE("Iterator - STL compatibility")
{
    SECTION("Matrix std::copy")
    {
        mtl::Matrix<int, 2, 2> matrix1{ 1, 2, 3, 4 };
        mtl::Matrix<int, 2, 2> matrix2;

        std::copy(matrix1.begin(), matrix1.end(), matrix2.begin());

        REQUIRE(matrix1 == matrix2);
    }

    SECTION("Matrix std::accumulate")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
        int sum = std::accumulate(matrix.begin(), matrix.end(), 0);

        REQUIRE(sum == 10);
    }

    SECTION("Matrix std::transform")
    {
        mtl::Matrix<int, 2, 2> matrix1{ 1, 2, 3, 4 };
        mtl::Matrix<int, 2, 2> matrix2;

        std::transform(
            matrix1.begin(),
            matrix1.end(),
            matrix2.begin(),
            [](int x) { return x * 2; });

        mtl::Matrix<int, 2, 2> expected{ 2, 4, 6, 8 };

        REQUIRE(matrix2 == expected);
    }

    SECTION("Matrix std::for_each")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
        int sum = 0;

        std::for_each(matrix.begin(), matrix.end(), [&sum](int x) {
            sum += x;
        });

        REQUIRE(sum == 10);
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
    SECTION("Same size transposition")
    {
        mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
        const mtl::Matrix<int, 2, 2> result{ 1, 3, 2, 4 };
        REQUIRE(matrix.transpose() == result);
    }

    SECTION("Different size transposition")
    {
        mtl::Matrix<double, 3, 2> matrix{ { 1, 2 }, { 3, 4 }, { 5, 6 } };
        const mtl::Matrix<double, 2, 3> result{ { 1, 3, 5 }, { 2, 4, 6 } };
        REQUIRE(matrix.transpose() == result);
    }
}

TEST_CASE("Determinant")
{
    SECTION("Matrix")
    {
        mtl::Matrix<int, 3, 3> matrix{ 5, 2, 3, 4, 5, 6, 7, 8, 9 };

        REQUIRE(matrix.det() == -12);
    }

    SECTION("const Matrix") {}

    SECTION("1x1 Matrix")
    {
        mtl::Matrix<int, 1, 1> matrix{ 1 };
        REQUIRE(matrix.det() == 1);
    }

    SECTION("2x2 Matrix")
    {
        mtl::Matrix<int, 2, 2> matrix{ 2, 3, 4, 5 };
        REQUIRE(matrix.det() == -2);
    }

    SECTION("incorrect Matrix size")
    {
        mtl::Matrix<int, 2, 3> matrix{};
        REQUIRE_THROWS_AS(matrix.det(), std::logic_error);
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

TEST_CASE("Addition")
{
    SECTION("Addition Same Size")
    {
        const mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 2, 3, 4, 5 };
        const mtl::Matrix<double, 2, 2> result{ 3, 5, 7, 9 };

        REQUIRE(matrix1 + matrix2 == result);
    }

    SECTION("Addition Different Sizes")
    {
        const mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 3, 3> matrix2{ 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        REQUIRE_THROWS_AS(matrix1 + matrix2, std::logic_error);
    }

    SECTION("Addition Different Types")
    {
        const mtl::Matrix<int, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 2.5, 3.5, 4.5, 5.5 };

        const mtl::Matrix<double, 2, 2> result{ 3.5, 5.5, 7.5, 9.5 };

        REQUIRE(matrix1 + matrix2 == result);
    }
}

TEST_CASE("Addition Assignment ")
{
    SECTION("Same Size")
    {
        mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 2, 3, 4, 5 };
        const mtl::Matrix<double, 2, 2> result{ 3, 5, 7, 9 };

        matrix1 += matrix2;

        REQUIRE(matrix1 == result);
    }

    SECTION("Addition Assignment Different Types")
    {
        mtl::Matrix<int, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 2.5, 3.5, 4.5, 5.5 };
        const mtl::Matrix<int, 2, 2> result{ 3, 5, 7, 9 };

        matrix1 += matrix2;

        REQUIRE(matrix1 == result);
    }

    SECTION("Addition Assignment Different Sizes")
    {
        mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 3, 3> matrix2{ 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        REQUIRE_THROWS_AS(matrix1 += matrix2, std::logic_error);
    }
}

TEST_CASE("Subtraction and Subtraction Assignment")
{
    SECTION("Subtraction Same Size")
    {
        const mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 2, 1, 4, 3 };
        const mtl::Matrix<double, 2, 2> result{ -1, 1, -1, 1 };

        REQUIRE(matrix1 - matrix2 == result);
    }

    SECTION("Subtraction Different Types")
    {
        const mtl::Matrix<int, 2, 2> matrix1{ 5, 5, 5, 5 };
        const mtl::Matrix<double, 2, 2> matrix2{ 1.5, 2.5, 3.5, 4.5 };
        const mtl::Matrix<double, 2, 2> result{ -3.5, -2.5, -1.5, -0.5 };

        REQUIRE(matrix2 - matrix1 == result);
    }

    SECTION("Subtraction Different Sizes")
    {
        const mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 3, 3> matrix2{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        REQUIRE_THROWS_AS(matrix1 - matrix2, std::logic_error);
    }
}

TEST_CASE("Subtraction Assignment")
{
    SECTION("Subtraction Assignment Same Size")
    {
        mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 2, 1, 4, 3 };
        const mtl::Matrix<double, 2, 2> result{ -1, 1, -1, 1 };

        matrix1 -= matrix2;

        REQUIRE(matrix1 == result);
    }

    SECTION("Subtraction Assignment Different Types")
    {
        mtl::Matrix<int, 2, 2> matrix1{ 5, 5, 5, 5 };
        const mtl::Matrix<double, 2, 2> matrix2{ 1.5, 2.5, 3.5, 4.5 };
        const mtl::Matrix<int, 2, 2> result{ 4, 3, 2, 1 };

        matrix1 -= matrix2;

        REQUIRE(matrix1 == result);
    }

    SECTION("Subtraction Assignment Different Sizes")
    {
        mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 3, 3> matrix2{ 1, 1, 1, 1, 1, 1, 1, 1, 1 };

        REQUIRE_THROWS_AS(matrix1 -= matrix2, std::logic_error);
    }
}

TEST_CASE("Multiplication")
{
    SECTION("Multiplication Same Size")
    {
        const mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 2, 0, 1, 3 };
        const mtl::Matrix<double, 2, 2> result{ 4, 6, 10, 12 };

        REQUIRE(matrix1 * matrix2 == result);
    }

    SECTION("Multiplication Different Types")
    {
        const mtl::Matrix<int, 2, 2> matrix1{ 2, 1, 3, 4 };
        const mtl::Matrix<double, 2, 2> matrix2{ 1.5, 2.5, 3.5, 4.5 };
        const mtl::Matrix<double, 2, 2> result{ 6.5, 9.5, 18.5, 25.5 };

        REQUIRE(matrix1 * matrix2 == result);
    }

    SECTION("Multiplication Different Sizes -> 2x2")
    {
        const mtl::Matrix<double, 2, 3> matrix1{ 1, 2, 3, 4, 5, 6 };
        const mtl::Matrix<double, 3, 2> matrix2{ 2, 0, 1, 3, 5, 2 };
        const mtl::Matrix<double, 2, 2> result{ 19, 12, 43, 27 };

        REQUIRE(matrix1 * matrix2 == result);
    }

    SECTION("Multiplication Different Sizes -> 3x3")
    {
        const mtl::Matrix<int, 3, 2> matrix1{ 1, 2, 3, 4, 5, 6 };
        const mtl::Matrix<int, 2, 3> matrix2{ 1, 2, 3, 4, 5, 6 };
        const auto result_values = { 9, 12, 15, 19, 26, 33, 29, 40, 51 };
        const auto result = matrix1 * matrix2;

        REQUIRE(result.size() == std::pair<std::size_t, std::size_t>{ 3, 3 });
        REQUIRE(result == result_values);
    }
}

TEST_CASE("Scalar Multiplication")
{
    SECTION("Scalar Multiplication Same Size")
    {
        const mtl::Matrix<double, 2, 2> matrix1{ 1, 2, 3, 4 };
        const double scalar = 2.5;
        const mtl::Matrix<double, 2, 2> result{ 2.5, 5, 7.5, 10 };

        REQUIRE(matrix1 * scalar == result);
    }

    SECTION("Scalar Multiplication Different Types")
    {
        const mtl::Matrix<int, 2, 2> matrix2{ 2, 1, 3, 4 };
        const double scalar = 1.5;
        const mtl::Matrix<double, 2, 2> result{ 3, 1.5, 4.5, 6 };

        REQUIRE(matrix2 * scalar == result);
    }

    SECTION("Scalar Multiplication Different Sizes")
    {
        const mtl::Matrix<double, 2, 3> matrix3{ 1, 2, 3, 4, 5, 6 };
        const double scalar = 0.5;
        const mtl::Matrix<double, 2, 3> result{ 0.5, 1, 1.5, 2, 2.5, 3 };

        REQUIRE(matrix3 * scalar == result);
    }
}

TEST_CASE("Multiplication - quadratic matrix")
{
    const mtl::Matrix<int, 2, 2> matrix{ 1, 2, 3, 4 };
    const auto result_values = { 7, 10, 15, 22 };

    auto result = matrix * matrix;

    REQUIRE(result.size() == matrix.size());
    REQUIRE(result == result_values);
}