#ifndef MTL_MATRIX_HPP
#define MTL_MATRIX_HPP

#include <concepts>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace mtl {

namespace detail {

template <class At_>
static inline constexpr auto is_arithmetic_v = std::is_arithmetic<At_>::value;

template <class At_, class Au_>
static inline constexpr auto is_same_v = std::is_same<At_, Au_>::value;

template <class At_, class Au_>
static inline constexpr auto is_convertible_v =
    std::is_convertible<Au_, Au_>::value;

namespace exceptions {

struct out_of_range_input : public std::out_of_range {
    using out_of_range::out_of_range;
};

struct invalid_argument_input : public std::invalid_argument {
    using invalid_argument::invalid_argument;
};

}  // namespace exceptions

// clang-format off
template <typename At_>
concept Arithmetic = is_arithmetic_v<At_> and requires(At_ type)
{
    type + type;
    type - type;
    type * type;
    type == type;
    type != type;
};

template <class At_, class Au_>
concept Scalar = std::is_scalar_v<Au_> and requires(At_ t, Au_ u)
{
    t * u;
};
// clang-format on

}  // namespace detail

/**
 * @brief Helper class for storing entire row from matrix
 */
template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Row;

/**
 * @brief Helper class for storing entire const row from const matrix
 */
template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Crow;

/**
 * @brief Main class for matrix storage
 * @tparam T type of value's being stored
 * @tparam I number of rows
 * @tparam J number of cols
 */
template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Matrix {
   private:
    T** array{nullptr};
    std::pair<std::size_t, std::size_t> size_{I, J};
    bool has_been_reallocated{false};

   public:
    /**
     * @brief Default constructor of mtl::Matrix<T, I, J>
     * @see Matrix(const T& value)
     */
    constexpr Matrix();

    /**
     * @brief Default destructor of mtl::Matrix<T, I, J>
     */
    constexpr ~Matrix();

    /**
     * @brief Constructor with initialization matrix with custom value
     * @param value initializator of matrix
     */
    explicit Matrix(const T& value);

    /**
     * @brief Constructor for initializer list
     * @param elems initializer list of elements
     */
    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    Matrix(std::initializer_list<T> elems);

    /**
     * @brief Constructor for initializer list
     * @param elems initializer list of elements
     */
    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    Matrix(std::initializer_list<std::initializer_list<T>> elems);

    /**
     * @brief Constructor with initialization matrix with custom value with
     * casting to base type
     * @param value initializator of matrix
     */
    template <detail::Arithmetic U>
    explicit Matrix(const U& value);

    /**
     * @brief Copy constructor
     * @param array matrix to copy
     */
    Matrix(const Matrix<T, I, J>& array);

    /**
     * @brief Move constructor
     * @param array matrix to move
     */
    Matrix(Matrix<T, I, J>&& matrix) noexcept;

    /**
     * @brief Copy constructor for matrix with different template args
     * @param array matrix to copy
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    explicit Matrix(const Matrix<U, A, B>& matrix);

    /**
     * @brief Move constructor for matrix with different template args
     * @param array matrix to move
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    explicit Matrix(Matrix<U, A, B>&& array) noexcept;

    /**
     * @brief Copy assignment operator
     * @param matrix matrix to copy
     * @return Reference to base matrix
     */
    constexpr auto operator=(const Matrix<T, I, J>& matrix) -> Matrix<T, I, J>&;

    /**
     * @brief Move assignment operator
     * @param matrix matrix to move
     * @return Reference to base matrix
     */
    constexpr auto operator=(Matrix<T, I, J>&& matrix) noexcept
        -> Matrix<T, I, J>&;

    /**
     * @brief Copy assignment operator for matrix with different template args
     * @param matrix matrix to copy
     * @return Reference to base matrix
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator=(const Matrix<U, A, B>& matrix) -> Matrix<T, I, J>&;

    /**
     * @brief Move assignment operator for matrix with different template args
     * @param matrix matrix to move
     * @return Reference to base matrix
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator=(Matrix<U, A, B>&& matrix) noexcept
        -> Matrix<T, I, J>&;

    /**
     * @brief Get underlying array
     * @return underlying array pointer
     */
    constexpr auto underlying_array() -> T**;

    /**
     * @brief Get underlying array for const object
     * @return underlying array pointer
     */
    constexpr auto underlying_array() const -> T**;

    /**
     * @brief Fill matrix with certain value
     * @param element value to fill
     */
    auto insert(const T& element);

    /**
     * @brief
     */
    auto sort();

    /**
     * @brief method transposes the matrix
     * @return transposed matrix
     */
    auto transpose() -> Matrix<T, I, J>;

    /**
     * @brief method transposes the matrix
     * @return transposed matrix
     */
    auto transpose() const -> Matrix<T, I, J>;

    /**
     * @brief method powers the matrix
     * @param power integer of the power to calc
     * @return powered matrix
     */
    auto power(unsigned int power) -> Matrix<T, I, J>&;

    /**
     * @brief methods calculates determinant of the matrix
     * @return det
     */
    auto det() -> T;

    /**
     * @brief methods calculates determinant of the matrix
     * @return det
     */
    auto det() const -> T;

    /**
     * @brief check whether the matrix is diagonal
     * @return is diagonal
     */
    auto is_diagonal() -> bool;

    /**
     * @brief check whether the matrix is diagonal
     * @return is diagonal
     */
    auto is_diagonal() const -> bool;

    /**
     * @brief fill matrix with "1"
     */
    auto ones();

    /**
     * @brief fill matrix with "1"
     */
    auto ones() const;

    auto alloc();
    auto alloc() const;
    auto alloc(std::size_t, std::size_t);
    auto alloc(std::size_t, std::size_t) const;
    auto realloc(std::size_t, std::size_t);

    auto dealloc();
    auto dealloc() const;

    /**
     * @brief get size i (rows) and j (cols) of matrix
     * @return size: pair of i and j
     */
    constexpr auto size() -> std::pair<std::size_t, std::size_t>;

    /**
     * @brief get size i (rows) and j (cols) of matrix
     * @return size: pair of i and j
     */
    constexpr auto size() const -> std::pair<std::size_t, std::size_t>;

    /**
     * @brief get size i (number of rows)
     * @return number of rows
     */
    constexpr auto size_i() -> std::size_t;

    /**
     * @brief get size i (number of rows)
     * @return number of rows
     */
    constexpr auto size_i() const -> std::size_t;

    /**
     * @brief get size j (number of cols)
     * @return number of cols
     */
    constexpr auto size_j() -> std::size_t;

    /**
     * @brief get size j (number of cols)
     * @return number of cols
     */
    constexpr auto size_j() const -> std::size_t;

    /**
     * @brief operator+= overload
     * @param matrix to add
     * @return reference to result of adding two matrices
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    auto operator+=(const Matrix<U, A, B>& matrix) -> Matrix<T, I, J>&;

    /**
     * @brief operator-= overload
     * @param matrix to subtract
     * @return reference to result of subtraction two matrices
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    auto operator-=(const Matrix<U, A, B>& matrix) -> Matrix<T, I, J>&;

    /**
     * @brief operator*= overload
     * @param matrix to multiply
     * @return result of multiplication two matrices
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    auto operator*=(const Matrix<U, A, B>& matrix) -> Matrix<T, I, B>&;

    /**
     * @brief operator*= overload
     * @param scalar to multiply
     * @return result of multiplication of matrix and scalar
     */
    template <detail::Scalar<T> U>
    auto operator*=(const U& scalar) -> Matrix<T, I, J>&;

    /**
     * @brief operator*= overload
     * @param vector to multiply
     * @return result of multiplication of matrix and vector
     */
    template <detail::Scalar<T> U>
    auto operator*=(const std::vector<U>& vector) -> Matrix<T, I, J>&;

    /**
     * @brief operator^ overload
     * @param power integer of the power to calc
     * @return powered matrix
     */
    auto operator^(const unsigned int& power) -> Matrix<T, I, J>&;

    /**
     * @brief operator== overload
     * @param matrix matrix to compare
     * @return are two matrices equal
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    inline auto operator==(const Matrix<U, A, B>& matrix) const -> bool;

    /**
     * @brief operator== overload
     * @param matrix matrix to compare
     * @return are two matrices unequal
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    inline auto operator!=(const Matrix<U, A, B>& matrix) const -> bool;

    /**
     * @brief operator[] overload
     * @param row number of row
     * @return row with certain index
     */
    auto operator[](std::size_t row) -> Row<T, I, J>;

    /**
     * @brief operator[] overload
     * @param row number of row
     * @return row with certain index
     */
    auto operator[](std::size_t row) const -> Crow<T, I, J>;

    /**
     * @brief operator() overload
     * @param row number of row
     * @param col number of col
     * @return element of entire index (row, col)
     */
    auto operator()(std::size_t row, std::size_t col) -> T&;

    /**
     * @brief operator() overload
     * @param row number of row
     * @param col number of col
     * @return element of entire index (row, col)
     */
    auto operator()(std::size_t row, std::size_t col) const -> T;

    /**
     * @brief operator<< overload
     * @param os std::ostream output
     * @param array matrix to display
     * @return std::ostream output
     */
    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    friend auto operator<<(std::ostream& os, const Matrix<U, A, B>& array)
        -> std::ostream&;

    class iterator {
       private:
        Matrix<T, I, J>& matrix;
        std::size_t row;
        std::size_t col;

       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        iterator(Matrix<T, I, J>& matrix, std::size_t row, std::size_t col);
        auto operator*() -> T;
        auto operator++() -> iterator&;
        auto operator++(int) -> iterator;
        auto operator==(const iterator& iter) const -> bool;
        auto operator!=(const iterator& iter) const -> bool;
        friend class Matrix;
    };

    class const_iterator {
       private:
        const Matrix<T, I, J>& matrix;
        std::size_t row;
        std::size_t col;

       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        const_iterator(
            const Matrix<T, I, J>& matrix,
            std::size_t row,
            std::size_t col);
        auto operator*() const -> T;
        auto operator++() -> const_iterator&;
        auto operator++(int) -> const_iterator;
        auto operator==(const const_iterator& iter) const -> bool;
        auto operator!=(const const_iterator& iter) const -> bool;
        friend class Matrix;
    };

    class reverse_iterator {
       private:
        Matrix<T, I, J>& matrix;
        std::size_t row;
        std::size_t col;

       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        reverse_iterator(
            Matrix<T, I, J>& matrix,
            std::size_t row,
            std::size_t col);
        auto operator*() const -> const T&;
        auto operator*() -> T&;
        auto operator++() -> reverse_iterator&;
        auto operator++(int) -> reverse_iterator;
        auto operator==(const reverse_iterator& iter) const -> bool;
        auto operator!=(const reverse_iterator& iter) const -> bool;
        friend class Matrix;
    };

    class const_reverse_iterator {
       private:
        Matrix<T, I, J>& matrix;
        std::size_t row;
        std::size_t col;

       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = T*;
        using reference = T&;

        const_reverse_iterator(
            Matrix<T, I, J>& matrix,
            std::size_t row,
            std::size_t col);
        auto operator*() const -> const T&;
        auto operator*() -> T&;
        auto operator++() -> const_reverse_iterator&;
        auto operator++(int) -> const_reverse_iterator;
        auto operator==(const const_reverse_iterator& iter) const -> bool;
        auto operator!=(const const_reverse_iterator& iter) const -> bool;
        friend class Matrix;
    };

    /**
     * @brief begin of container
     * @return iterator on beginning
     */
    auto begin() -> iterator;

    /**
     * @brief begin of container
     * @return const_iterator on beginning
     */
    auto begin() const -> const_iterator;

    /**
     * @brief end of container
     * @return iterator on end
     */
    auto end() -> iterator;

    /**
     * @brief begin of container
     * @return iterator on end
     */
    auto end() const -> const_iterator;

    auto cbegin() -> const_iterator;
    auto rbegin() -> reverse_iterator;
    auto crbegin() -> const_reverse_iterator;
    auto cbegin() const -> const_iterator;
    auto rbegin() const -> reverse_iterator;
    auto crbegin() const -> const_reverse_iterator;
    auto cend() -> const_iterator;
    auto rend() -> reverse_iterator;
    auto crend() -> const_reverse_iterator;
    auto cend() const -> const_iterator;
    auto rend() const -> reverse_iterator;
    auto crend() const -> const_reverse_iterator;
};

template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Row {
   private:
    Matrix<T, I, J>& matrix;
    std::vector<T> row;
    std::size_t n_row;

   public:
    explicit Row(Matrix<T, I, J>& matrix, std::size_t n_row);
    ~Row() = default;
    Row(const Row& row) = default;
    Row(Row&& row) noexcept = default;
    constexpr auto operator=(const Row& row) -> Row&;
    constexpr auto operator=(Row&& row) noexcept -> Row&;

    auto operator[](std::size_t col) -> T&;
    auto operator[](std::size_t col) const -> T&;

    auto get_row() -> std::vector<T>&;
    auto get_row() const -> const std::vector<T>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    friend auto operator<<(std::ostream& os, const Row& row) -> std::ostream&;
};

template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Crow {
   private:
    const Matrix<T, I, J>& matrix;
    std::vector<T> row;
    std::size_t n_row;

   public:
    explicit Crow(const Matrix<T, I, J>& matrix, std::size_t n_row);
    ~Crow() = default;
    Crow(const Crow& row) = default;
    Crow(Crow&& row) noexcept = default;
    constexpr auto operator=(const Crow& row) -> Crow&;
    constexpr auto operator=(Crow&& row) noexcept -> Crow&;

    auto operator[](std::size_t col) -> const T&;
    auto operator[](std::size_t col) const -> const T&;

    auto get_row() -> std::vector<T>&;
    auto get_row() const -> const std::vector<T>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    friend auto operator<<(std::ostream& os, const Crow& row) -> std::ostream&;
};

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix()
{
    array = new T*[I];
    for (std::size_t i = 0; i < I; ++i) { array[i] = new T[J]; }

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { this->array[i][j] = 0; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::~Matrix()
{
    if (not has_been_reallocated) {
        for (std::size_t i = 0; i < I; ++i) { delete[] array[i]; }
        delete[] array;
        array = nullptr;
    }
    else {
        for (std::size_t i = 0; i < size_.first; ++i) { delete[] array[i]; }
        delete[] array;
        array = nullptr;
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix(const T& value)
{
    this->alloc();
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { this->array[i][j] = value; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix(std::initializer_list<T> elems)
{
    this->alloc();
    this->ones();
    if (elems.size() != I * J) [[unlikely]] {
        std::cout << "Matrix::Matrix() cannot initialize matrix with incorrect "
                     "number of elements."
                  << std::endl;
    }
    else [[likely]] {
        auto i = 0;
        auto j = 0;
        for (const auto& elem : elems) {
            array[i][j] = elem;
            if (j != J - 1) { ++j; }
            else if (i != I - 1) {
                ++i;
                j = 0;
            }
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix(std::initializer_list<std::initializer_list<T>> elems)
{
    this->alloc();
    this->ones();
    for (const auto& elem : elems) {
        if (elems.size() * elem.size() != I * J) [[unlikely]] {
            std::cout
                << "Matrix::Matrix() cannot initialize matrix with incorrect "
                   "number of elements."
                << std::endl;
            return;
        }
    }

    auto i = 0;
    auto j = 0;
    for (const auto& row : elems) {
        for (const auto& item : row) {
            array[i][j] = item;
            if (j != J - 1) { ++j; }
            else if (i != I - 1) {
                ++i;
                j = 0;
            }
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U>
Matrix<T, I, J>::Matrix(const U& value)
{
    this->alloc();

    static_assert(
        std::is_convertible_v<T, U>,
        "Matrix::Matrix(), type cannot be used to initialize matrix");

    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            this->array[i][j] = static_cast<T>(value);
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix(const Matrix<T, I, J>& array)
{
    this->alloc();

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            this->array[i][j] = array.array[i][j];
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
Matrix<T, I, J>::Matrix(const Matrix<U, A, B>& matrix)
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A and J == B, "Matrix::invalid size");
    *this = matrix;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix(Matrix<T, I, J>&& matrix) noexcept
{
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            this->array[i][j] = matrix.array[i][j];
        }
    }

    matrix.~Matrix();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
Matrix<T, I, J>::Matrix(Matrix<U, A, B>&& array) noexcept
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A and J == B, "Matrix::invalid size");

    *this = array;

    array.~Matrix();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(const Matrix<T, I, J>& matrix)
    -> Matrix<T, I, J>&
{
    if constexpr (&matrix != this) {
        for (auto i = 0; i < I; ++i) {
            for (auto j = 0; j < J; ++j) {
                this->array[i][j] = matrix.array[i][j];
            }
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A and J == B, "Matrix::invalid size");

    *this = matrix;

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(Matrix<T, I, J>&& matrix) noexcept
    -> Matrix<T, I, J>&
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = matrix.array[i][j]; }
    }

    matrix.~Matrix();

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(Matrix<U, A, B>&& matrix) noexcept
    -> Matrix<T, I, J>&
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A and J == B, "Matrix::invalid size");

    *this = matrix;

    matrix.~Matrix();

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::underlying_array() -> T**
{
    return array;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::underlying_array() const -> T**
{
    return array;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::insert(const T& element)
{
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { this->array[i][j] = element; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::sort()
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::transpose() -> Matrix<T, I, J>
{
    Matrix<T, I, J> result;
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            result.array[i][j] = this->array[j][i];
        }
    }

    return result;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::transpose() const -> Matrix<T, I, J>
{
    Matrix<T, I, J> result;
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            result.array[i][j] = this->array[j][i];
        }
    }

    return result;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::det() -> T
{
    static_assert(I == J, "Matrix::det::invalid size");

    T det = 0;
    int sign = -1;

    Matrix temp;

    if constexpr (I == 1) { return this->array[0][0]; }

    if constexpr (I == 2) {
        return (
            this->array[0][0] * this->array[1][1]
            - this->array[0][1] * this->array[1][0]);
    }

    for (auto m = 0; m < I; ++m) {
        unsigned int temp_i = 0;
        unsigned int temp_j = 0;
        for (auto n = 1; n < I; ++n) {
            for (auto o = 0; o < I; ++o) {
                if (o == m) { continue; }
                temp.array[temp_i][temp_j] = this->array[n][o];
                temp_j++;
            }
            if (temp_j == (I - 1)) {
                temp_i++;
                temp_j = 0;
            }
        }
        sign = -sign;
        det += (sign * this->array[0][m] * temp.det());
    }

    return det;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::det() const -> T
{
    static_assert(I == J, "Matrix::det::invalid size");

    T det = 0;
    int sign = -1;

    Matrix temp;

    if constexpr (I == 1) { return this->array[0][0]; }

    if constexpr (I == 2) {
        return (
            this->array[0][0] * this->array[1][1]
            - this->array[0][1] * this->array[1][0]);
    }

    for (auto m = 0; m < I; ++m) {
        unsigned int temp_i = 0;
        unsigned int temp_j = 0;
        for (auto n = 1; n < I; ++n) {
            for (auto o = 0; o < I; ++o) {
                if (o == m) { continue; }
                temp.array[temp_i][temp_j] = this->array[n][o];
                temp_j++;
            }
            if (temp_j == (I - 1)) {
                temp_i++;
                temp_j = 0;
            }
        }
        sign = -sign;
        det += (sign * this->array[0][m] * temp.det());
    }

    return det;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
[[nodiscard]] auto Matrix<T, I, J>::is_diagonal() -> bool
{
    if constexpr (I != J) { return false; }

    for (std::size_t m = 0; m < I; ++m) {
        for (std::size_t n = 0; n < I; ++n) {
            if (m != n and this->array[m][n] != 0) { return false; }
        }
    }

    return true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
[[nodiscard]] auto Matrix<T, I, J>::is_diagonal() const -> bool
{
    if constexpr (I != J) { return false; }

    for (std::size_t m = 0; m < I; ++m) {
        for (std::size_t n = 0; n < I; ++n) {
            if (m != n and this->array[m][n] != 0) { return false; }
        }
    }

    return true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::ones()
{
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { array[i][j] = 0; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::ones() const
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { *this(i, j) = 0; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::alloc()
{
    array = new T*[I];
    for (std::size_t i = 0; i < I; ++i) { array[i] = new T[J]; }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::alloc() const
{
    array = new T*[I];
    for (std::size_t i = 0; i < I; ++i) { array[i] = new T[J]; }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::alloc(std::size_t i_, std::size_t j_)
{
    array = new T*[i_];
    for (std::size_t i = 0; i < i_; ++i) { array[i] = new T[j_]; }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::alloc(std::size_t i_, std::size_t j_) const
{
    array = new T*[i_];
    for (std::size_t i = 0; i < i_; ++i) { array[i] = new T[j_]; }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::realloc(std::size_t i_, std::size_t j_)
{
    for (std::size_t i = 0; i < size_.first; ++i) { delete[] array[i]; }
    delete[] array;
    array = nullptr;

    array = new T*[i_];
    for (std::size_t i = 0; i < i_; ++i) { array[i] = new T[j_]; }

    has_been_reallocated = true;
    size_ = std::make_pair(i_, j_);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size() -> std::pair<std::size_t, std::size_t>
{
    return size_;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size() const
    -> std::pair<std::size_t, std::size_t>
{
    return size_;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size_i() -> std::size_t
{
    return size_.first;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size_i() const -> std::size_t
{
    return size_.first;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size_j() -> std::size_t
{
    return size_.second;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size_j() const -> std::size_t
{
    return size_.second;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::power(unsigned int power) -> Matrix<T, I, J>&
{
    for (unsigned int i = 0; i < (power - 1); ++i) { *this *= *this; }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator+=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(I == A and J == B, "Matrix::invalid size");
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            this->array[i][j] += matrix.array[i][j];
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator-=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(I == A and J == B, "Matrix::invalid size");
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            this->array[i][j] -= matrix.array[i][j];
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator*=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, B>&
{
    static_assert(J == A, "Matrix::invalid size");
    static_assert(detail::is_convertible_v<T, U>, "Matrix::invalid type");

    // realloc needs to be done on this
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < B; ++j) {
            this->array[i][j] = 0;
            for (auto k = 0; k < B; ++k) {
                this->array[i][j] += this->array[i][k] * matrix.array[k][j];
            }
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Scalar<T> U>
auto Matrix<T, I, J>::operator*=(const U& scalar) -> Matrix<T, I, J>&
{
    for (std::size_t i = 0; i < J; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            this->array[i][j] = this->array[i][j] * scalar;
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Scalar<T> U>
auto Matrix<T, I, J>::operator*=(const std::vector<U>& vector)
    -> Matrix<T, I, J>&
{
    // constexpr auto mult_col_dim = 1;
    static_assert(J == vector.size(), "Matrix::invalid vector size");
    static_assert(detail::is_convertible_v<T, U>, "Matrix::invalid type");
    const auto temp = *this;

    // reallocation to <T, J, 1>
    // this->realloc(J, 1);

    for (std::size_t i = 0; i < I; ++i) {
        array[i][0] = 0;
        for (std::size_t j = 0; j < J; ++j) {
            array[i][0] += temp[i][j] * vector[j];
        }
    }

    return *this;
}

template <
    detail::Arithmetic T,
    detail::Arithmetic U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
auto operator+(Matrix<T, I, J> lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<T, I, J>
{
    lhs += rhs;
    return lhs;
}

template <
    detail::Arithmetic T,
    detail::Arithmetic U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
auto operator-(Matrix<T, I, J> lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<T, I, J>
{
    lhs -= rhs;
    return lhs;
}

template <
    detail::Arithmetic T,
    detail::Arithmetic U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
auto operator*(Matrix<T, I, J> lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<T, I, J>
{
    lhs *= rhs;
    return lhs;
}

template <
    detail::Arithmetic T,
    detail::Scalar<T> U,
    std::size_t I,
    std::size_t J>
auto operator*(Matrix<T, I, J> lhs, const U& rhs) -> Matrix<T, I, J>
{
    lhs *= rhs;
    return lhs;
}

template <
    detail::Arithmetic T,
    detail::Scalar<T> U,
    std::size_t I,
    std::size_t J>
auto operator*(U lhs, const Matrix<T, I, J>& rhs) -> Matrix<T, I, J>
{
    const_cast<Matrix<T, I, J>&>(rhs) *= lhs;
    return rhs;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator^(const unsigned int& power) -> Matrix<T, I, J>&
{
    this->power(power);

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
inline auto Matrix<T, I, J>::operator==(const Matrix<U, A, B>& matrix) const
    -> bool
{
    if constexpr (I != A or J != B or not detail::is_convertible_v<T, U>) {
        return false;
    }

    for (std::size_t i = 0; i < J; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            if (this->array[i][j] != matrix.array[i][j]) { return false; }
        }
    }

    return true;
}

// clang-format off
template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
inline auto Matrix<T, I, J>::operator!=(const Matrix<U, A, B>& matrix) const
    -> bool
{
    return not (*this == matrix);
}
// clang-format on

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator[](std::size_t row) -> Row<T, I, J>
{
    if (row > (I - 1) or row < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"};
    }
    return Row(*this, row);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator[](std::size_t row) const -> Crow<T, I, J>
{
    if (row > (I - 1) or row < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"};
    }
    return Crow(*this, row);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Row<T, I, J>::Row(Matrix<T, I, J>& matrix, std::size_t n_row)
    : matrix{matrix}, n_row{n_row}
{
    for (std::size_t i = 0; i < J; ++i) { row.emplace_back(matrix(n_row, i)); }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Row<T, I, J>::operator[](std::size_t col) -> T&
{
    if (col > (J - 1) or col < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"};
    }
    return matrix(n_row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Row<T, I, J>::operator[](std::size_t col) const -> T&
{
    if (col > (J - 1) or col < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"};
    }
    return matrix(n_row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Row<T, I, J>::get_row() -> std::vector<T>&
{
    return row;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Row<T, I, J>::get_row() const -> const std::vector<T>&
{
    return row;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Crow<T, I, J>::Crow(const Matrix<T, I, J>& matrix, std::size_t n_row)
    : matrix{matrix}, n_row{n_row}
{
    for (auto i = 0; i < J; ++i) { row.emplace_back(matrix(n_row, i)); }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Crow<T, I, J>::operator[](std::size_t col) -> const T&
{
    if (col > (J - 1) or col < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"};
    }
    return matrix(n_row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Crow<T, I, J>::operator[](std::size_t col) const -> const T&
{
    if (col > (J - 1) or col < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"};
    }
    return matrix(n_row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Crow<T, I, J>::get_row() -> std::vector<T>&
{
    return row;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Crow<T, I, J>::get_row() const -> const std::vector<T>&
{
    return row;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col) -> T&
{
    if (row > (I - 1) or row < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"};
    }
    if (col > (J - 1) or col < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"};
    }

    return this->array[row][col];
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col) const -> T
{
    if (row > (I - 1) or row < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"};
    }
    if (col > (J - 1) or col < 0) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"};
    }

    return this->array[row][col];
}

template <detail::Arithmetic U, std::size_t A, std::size_t B>
auto operator<<(std::ostream& os, const Matrix<U, A, B>& array) -> std::ostream&
{
    for (std::size_t i = 0; i < A; ++i) {
        for (std::size_t j = 0; j < B; ++j) { os << array.array[i][j] << " "; }
        os << std::endl;
    }

    return os;
}

template <detail::Arithmetic U, std::size_t A, std::size_t B>
auto operator<<(std::ostream& os, const Row<U, A, B>& row) -> std::ostream&
{
    for (const auto& elem : row.get_row()) {
        os << elem;
        os << " ";
    }

    return os;
}

template <detail::Arithmetic U, std::size_t A, std::size_t B>
auto operator<<(std::ostream& os, const Crow<U, A, B>& row) -> std::ostream&
{
    for (const auto& elem : row.get_row()) {
        os << elem;
        os << " ";
    }

    return os;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::iterator::iterator(
    Matrix<T, I, J>& matrix_,
    std::size_t row_,
    std::size_t col_)
    : matrix{matrix_}, row{row_}, col{col_}
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::const_iterator::const_iterator(
    const Matrix<T, I, J>& matrix_,
    std::size_t row_,
    std::size_t col_)
    : matrix{matrix_}, row{row_}, col{col_}
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator*() -> T
{
    if (row > I - 1 or col > J - 1) { return static_cast<T>(0); }
    return matrix(row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator*() const -> T
{
    if (row > I - 1 and col > J - 1) { return static_cast<T>(0); }
    return matrix(row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++() -> Matrix<T, I, J>::iterator&
{
    if (col != J - 1) { ++col; }
    else {
        ++row;
        col = 0;
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator++()
    -> Matrix<T, I, J>::const_iterator&
{
    if (col != J - 1) { ++col; }
    else {
        ++row;
        col = 0;
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++(int) -> Matrix<T, I, J>::iterator
{
    auto temp = *this;

    ++(*this);

    return temp;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator++(int)
    -> Matrix<T, I, J>::const_iterator
{
    auto temp = *this;

    ++(*this);

    return temp;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator==(const iterator& iter) const -> bool
{
    return this->matrix == iter.matrix and row == iter.row and col == iter.col;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator==(
    const const_iterator& iter) const -> bool
{
    return this->matrix == iter.matrix and row == iter.row and col == iter.col;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator!=(const iterator& iter) const -> bool
{
    return not(*this == iter);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator!=(
    const const_iterator& iter) const -> bool
{
    return not(*this == iter);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::begin() -> Matrix<T, I, J>::iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::begin() const -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::end() -> Matrix<T, I, J>::iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::end() const -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::cbegin() -> Matrix<T, I, J>::const_iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::cbegin() const -> Matrix<T, I, J>::const_iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::cend() -> Matrix<T, I, J>::const_iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::cend() const -> Matrix<T, I, J>::const_iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::rbegin() -> Matrix<T, I, J>::reverse_iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::rbegin() const -> Matrix<T, I, J>::reverse_iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::rend() -> Matrix<T, I, J>::reverse_iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::rend() const -> Matrix<T, I, J>::reverse_iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::crbegin() -> Matrix<T, I, J>::const_reverse_iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::crbegin() const -> Matrix<T, I, J>::const_reverse_iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::crend() -> Matrix<T, I, J>::const_reverse_iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::crend() const -> Matrix<T, I, J>::const_reverse_iterator
{
    return iterator(*this, 0, 0);
}

}  // namespace mtl

#endif  // MTL_MATRIX_HPP