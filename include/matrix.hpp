//                     _          _         _  _  _
//                    | |        (_)       | |(_)| |
//   _ __ ___    __ _ | |_  _ __  _ __  __ | | _ | |__
//  | '_ ` _ \  / _` || __|| '__|| |\ \/ / | || || '_ |
//  | | | | | || (_| || |_ | |   | | >  <  | || || |_) |
//  |_| |_| |_| \__,_| \__||_|   |_|/_/\_\ |_||_||_.__/
//
//

#ifndef MTL_MATRIX_HPP
#define MTL_MATRIX_HPP

#include <array>
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
template <typename Ta_>
concept Arithmetic = is_arithmetic_v<Ta_> and requires(Ta_ a_type) {
                                                  a_type + a_type;
                                                  a_type - a_type;
                                                  a_type * a_type;
                                                  a_type == a_type;
                                                  a_type != a_type;
                                              };

template <class Ta_, class Tb_>
concept Scalar = std::is_scalar_v<Tb_>
                 and requires(Ta_ a_type, Tb_ b_type) { a_type * b_type; };
// clang-format on

}  // namespace detail

template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Row;

template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Crow;

// NOLINTBEGIN(hicpp-named-parameter,readability-named-parameter)
template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Matrix {
   private:
    T** array{ new T*[I] };
    std::pair<std::size_t, std::size_t> size_{ I, J };
    bool has_been_reallocated{ false };

   public:
    constexpr Matrix();

    constexpr ~Matrix();

    explicit constexpr Matrix(const T& value);

    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    constexpr Matrix(std::initializer_list<T> elems);

    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    constexpr Matrix(std::initializer_list<std::initializer_list<T>> elems);

    template <detail::Arithmetic U>
    explicit constexpr Matrix(const U& value);

    constexpr Matrix(const Matrix<T, I, J>& matrix);
    constexpr auto operator=(const Matrix<T, I, J>& matrix) -> Matrix<T, I, J>&;
    constexpr Matrix(Matrix<T, I, J>&& matrix) noexcept;
    constexpr auto operator=(Matrix<T, I, J>&& matrix) noexcept
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    explicit constexpr Matrix(const Matrix<U, A, B>& matrix);

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    explicit constexpr Matrix(Matrix<U, A, B>&& array) noexcept;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator=(const Matrix<U, A, B>& matrix) -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator=(Matrix<U, A, B>&& matrix) noexcept
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U>
    constexpr auto operator=(const std::initializer_list<U>& list)
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U>
    constexpr auto operator=(std::initializer_list<U>&& list)
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    explicit constexpr operator Matrix<U, A, B>() const;

    constexpr auto underlying_array() -> T**;

    constexpr auto underlying_array() const -> T**;

    constexpr auto insert(const T& element);

    constexpr auto sort();

    constexpr auto transpose() -> Matrix<T, J, I>;

    constexpr auto transpose() const -> Matrix<T, J, I>;

    constexpr auto power(unsigned int power) -> Matrix<T, I, J>&;

    constexpr auto det() -> T;

    constexpr auto det() const -> T;

    constexpr auto is_diagonal() -> bool;

    constexpr auto is_diagonal() const -> bool;

    constexpr auto ones();

    constexpr auto alloc();
    constexpr auto alloc() const;
    constexpr auto alloc(std::size_t, std::size_t);
    constexpr auto alloc(std::size_t, std::size_t) const;
    constexpr auto realloc(std::size_t, std::size_t);

    constexpr auto dealloc();
    constexpr auto dealloc() const;

    constexpr auto size() const noexcept -> std::pair<std::size_t, std::size_t>;

    constexpr auto size_i() const noexcept -> std::size_t;

    constexpr auto size_j() const noexcept -> std::size_t;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator+=(const Matrix<U, A, B>& matrix)
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator-=(const Matrix<U, A, B>& matrix)
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator*=(const Matrix<U, A, B>& matrix)
        -> Matrix<T, I, B>&;

    template <detail::Scalar<T> U>
    constexpr auto operator*=(const U& scalar) -> Matrix<T, I, J>&;

    template <detail::Scalar<T> U>
    constexpr auto operator*=(const std::vector<U>& vector) -> Matrix<T, I, J>&;

    constexpr auto operator^(const unsigned int& power) -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr inline auto operator==(const Matrix<U, A, B>& matrix) const
        -> bool;

    template <detail::Arithmetic U>
    constexpr inline auto operator==(const std::initializer_list<U>& list) const
        -> bool;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr inline auto operator!=(const Matrix<U, A, B>& matrix) const
        -> bool;

    constexpr auto operator[](std::size_t row) -> Row<T, I, J>;

    constexpr auto operator[](std::size_t row) const -> Crow<T, I, J>;

    constexpr auto operator()(std::size_t row, std::size_t col) -> T&;

    constexpr auto operator()(std::size_t row, std::size_t col) const
        -> const T&;

    constexpr auto at(std::size_t row, std::size_t col) const -> const T;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    friend constexpr auto operator<<(
        std::ostream& ostream,
        const Matrix<U, A, B>& array) -> std::ostream&;

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

    constexpr auto begin() -> iterator;
    constexpr auto end() -> iterator;

    constexpr auto begin() const -> const_iterator;
    constexpr auto end() const -> const_iterator;
};

template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Row {
   private:
    Matrix<T, I, J>& matrix;
    std::vector<T> row;
    std::size_t n_row;

   public:
    explicit Row(Matrix<T, I, J>&, std::size_t);
    ~Row() = default;
    Row(const Row&) = default;
    Row(Row&&) noexcept = default;
    constexpr auto operator=(const Row&) -> Row<T, I, J>& = default;
    constexpr auto operator=(Row&&) noexcept -> Row<T, I, J>& = default;
    constexpr auto operator=(const std::vector<T>&) -> Row<T, I, J>&;

    auto operator[](std::size_t) -> T&;

    auto get_row() const -> const std::vector<T>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    friend auto operator<<(std::ostream&, const Row&) -> std::ostream&;
};

template <detail::Arithmetic T, std::size_t I, std::size_t J>
class Crow {
   private:
    const Matrix<T, I, J>& matrix;
    std::vector<T> row;
    std::size_t n_row;

   public:
    explicit Crow(const Matrix<T, I, J>&, std::size_t);
    ~Crow() = default;
    Crow(const Crow&) = default;
    Crow(Crow&&) noexcept = default;
    constexpr auto operator=(const Crow&) -> Crow<T, I, J>& = default;
    constexpr auto operator=(Crow&&) noexcept -> Crow<T, I, J>& = default;

    auto operator[](std::size_t) const -> T;

    auto get_row() const -> const std::vector<T>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    friend auto operator<<(std::ostream&, const Crow&) -> std::ostream&;
};
// NOLINTEND(hicpp-named-parameter,readability-named-parameter)

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix()
{
    for (std::size_t i = 0; i < I; ++i) { array[i] = new T[J]; }

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { this->array[i][j] = 0; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::~Matrix()
{
    for (std::size_t i = 0; i < size_.first; ++i) { delete[] array[i]; }
    delete[] array;
    array = nullptr;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(const T& value)
{
    alloc();

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { this->array[i][j] = value; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(std::initializer_list<T> elems)
{
    alloc();

    ones();
    if (elems.size() != (I * J)) [[unlikely]] {
        throw detail::exceptions::invalid_argument_input{
            "Matrix::Matrix() cannot initialize matrix with incorrect "
            "number of elements."
        };
    }
    else [[likely]] {
        auto row_num = 0;
        auto col_num = 0;
        for (const auto& elem : elems) {
            array[row_num][col_num] = elem;
            if (col_num != J - 1) { ++col_num; }
            else if (row_num != I - 1) {
                ++row_num;
                col_num = 0;
            }
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(
    std::initializer_list<std::initializer_list<T>> elems)
{
    alloc();

    ones();

    for (const auto& elem : elems) {
        if (elems.size() * elem.size() != I * J) [[unlikely]] {
            throw detail::exceptions::invalid_argument_input{
                "Matrix::Matrix() cannot initialize matrix with incorrect "
                "number of elements."
            };
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
constexpr Matrix<T, I, J>::Matrix(const U& value)
{
    alloc();

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
constexpr Matrix<T, I, J>::Matrix(const Matrix<T, I, J>& matrix)
{
    *this = matrix;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(const Matrix<T, I, J>& matrix)
    -> Matrix<T, I, J>&
{
    if (&matrix != this) {
        alloc();

        for (std::size_t i = 0; i < I; ++i) {
            for (std::size_t j = 0; j < J; ++j) {
                this->array[i][j] = matrix.array[i][j];
            }
        }
    }
    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(Matrix<T, I, J>&& matrix) noexcept
    : size_{ 0, 0 }
{
    *this = std::move(matrix);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(Matrix<T, I, J>&& matrix) noexcept
    -> Matrix<T, I, J>&
{
    if (&matrix != this) {
        for (std::size_t i = 0; i < I; ++i) {
            for (std::size_t j = 0; j < J; ++j) {
                array[i][j] = std::move(matrix.array[i][j]);
            }
        }
        matrix.dealloc();
    }
    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr Matrix<T, I, J>::Matrix(const Matrix<U, A, B>& matrix)
{
    static_assert(detail::is_convertible_v<T, U>, "Matrix::invalid type");

    *this = matrix;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr Matrix<T, I, J>::Matrix(Matrix<U, A, B>&& array) noexcept
    : size_{ 0, 0 }
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A and J == B, "Matrix::invalid size");

    *this = std::move(array);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");

    *this = matrix;

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(Matrix<U, A, B>&& matrix) noexcept
    -> Matrix<T, I, J>&
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A and J == B, "Matrix::invalid size");

    *this = std::move(matrix);

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U>
constexpr auto Matrix<T, I, J>::operator=(const std::initializer_list<U>& list)
    -> Matrix<T, I, J>&
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid_type");

    if (I * J != list.size()) {
        throw detail::exceptions::invalid_argument_input{
            "Matrix::operator= std::initializer_list invalid size"
        };
    }

    auto row_num = 0;
    auto col_num = 0;
    for (const auto& elem : list) {
        array[row_num][col_num] = elem;
        if (col_num != J - 1) { ++col_num; }
        else if (row_num != I - 1) {
            ++row_num;
            col_num = 0;
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U>
constexpr auto Matrix<T, I, J>::operator=(std::initializer_list<U>&& list)
    -> Matrix<T, I, J>&
{
    static_assert(detail::is_same_v<T, U>, "Matrix::invalid_type");

    if (I * J != list.size()) {
        throw detail::exceptions::invalid_argument_input{
            "Matrix::operator= std::initializer_list invalid size"
        };
    }

    auto row_num = 0;
    auto col_num = 0;
    for (const auto& elem : list) {
        array[row_num][col_num] = std::move(elem);
        if (col_num != J - 1) { ++col_num; }
        else if (row_num != I - 1) {
            ++row_num;
            col_num = 0;
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
inline constexpr Matrix<T, I, J>::operator Matrix<U, A, B>() const
{
    static_assert(A >= I && B >= J, "Matrix::invalid size");

    Matrix<U, A, B> result;

    for (std::size_t i = 0; i < A; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            if (i < I and j < J) { result[i][j] = static_cast<U>(array[i][j]); }
            else {
                result[i][j] = U();
            }
        }
    }

    return result;
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
constexpr auto Matrix<T, I, J>::insert(const T& element)
{
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { this->array[i][j] = element; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::sort()
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::transpose() -> Matrix<T, J, I>
{
    Matrix<T, J, I> result{};
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            result.underlying_array()[j][i] = this->underlying_array()[i][j];
        }
    }

    return result;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::transpose() const -> Matrix<T, J, I>
{
    Matrix<T, J, I> result;
    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) { result[j][i] = this[i][j]; }
    }

    return result;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::det() -> T
{
    // static_assert(size_.first == size_.second, "Matrix::det::invalid size");

    T det = 0;
    int sign = -1;

    Matrix temp;

    if (size_.first == 1) { return array[0][0]; }

    if (size_.first == 2) {
        return (array[0][0] * array[1][1] - array[0][1] * array[1][0]);
    }

    for (std::size_t m = 0; m < size_.first; ++m) {
        std::size_t temp_i = 0;
        std::size_t temp_j = 0;
        for (std::size_t n = 1; n < size_.first; ++n) {
            for (std::size_t o = 0; o < size_.first; ++o) {
                if (o == m) { continue; }
                temp.array[temp_i][temp_j] = array[n][o];
                temp_j++;
            }
            if (temp_j == (I - 1)) {
                temp_i++;
                temp_j = 0;
            }
        }
        sign = -sign;
        det += (sign * array[0][m] * temp.det());
    }

    return det;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::det() const -> T
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
[[nodiscard]] constexpr auto Matrix<T, I, J>::is_diagonal() -> bool
{
    if constexpr (I != J) { return false; }

    for (std::size_t row_num = 0; row_num < I; ++row_num) {
        for (std::size_t col_num = 0; col_num < I; ++col_num) {
            if (row_num != col_num and array[row_num][col_num] != 0) {
                return false;
            }
        }
    }

    return true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
[[nodiscard]] constexpr auto Matrix<T, I, J>::is_diagonal() const -> bool
{
    if constexpr (I != J) { return false; }

    for (std::size_t row_num = 0; row_num < I; ++row_num) {
        for (std::size_t col_num = 0; col_num < I; ++col_num) {
            if (row_num != col_num and array[row_num][col_num] != 0) {
                return false;
            }
        }
    }

    return true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::ones()
{
    for (std::size_t i = 0; i < size_.first; ++i) {
        for (std::size_t j = 0; j < size_.second; ++j) { array[i][j] = 1; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc()
{
    array = new T*[size_.first];
    for (std::size_t i = 0; i < I; ++i) { array[i] = new T[size_.second]; }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc() const
{
    array = new T*[size_.first];
    for (std::size_t i = 0; i < I; ++i) { array[i] = new T[size_.second]; }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc(
    std::size_t row_size,
    std::size_t col_size)
{
    array = new T*[row_size];
    for (std::size_t i = 0; i < row_size; ++i) { array[i] = new T[col_size]; }

    ones();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc(
    std::size_t row_size,
    std::size_t col_size) const
{
    array = new T*[row_size];
    for (std::size_t i = 0; i < row_size; ++i) { array[i] = new T[col_size]; }

    ones();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::realloc(
    std::size_t row_size,
    std::size_t col_size)
{
    dealloc();

    size_ = std::make_pair(row_size, col_size);

    alloc(row_size, col_size);

    has_been_reallocated = true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::dealloc()
{
    for (std::size_t i = 0; i < size_.first; ++i) { delete[] array[i]; }
    delete[] array;
    array = nullptr;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::dealloc() const
{
    for (std::size_t i = 0; i < size_.first; ++i) { delete[] array[i]; }
    delete[] array;
    array = nullptr;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size() const noexcept
    -> std::pair<std::size_t, std::size_t>
{
    return size_;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size_i() const noexcept -> std::size_t
{
    return size_.first;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size_j() const noexcept -> std::size_t
{
    return size_.second;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::power(unsigned int power) -> Matrix<T, I, J>&
{
    const auto temp = *this;
    for (unsigned int i = 0; i < (power - 1); ++i) { *this *= temp; }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator+=(const Matrix<U, A, B>& matrix)
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
constexpr auto Matrix<T, I, J>::operator-=(const Matrix<U, A, B>& matrix)
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
constexpr auto Matrix<T, I, J>::operator*=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, B>&
{
    auto temp = *this;

    static_assert(J == A, "Matrix::invalid size");
    static_assert(detail::is_convertible_v<T, U>, "Matrix::invalid type");

    for (std::size_t i = 0; i < temp.size().first; ++i) {
        for (std::size_t j = 0; j < matrix.size().second; ++j) {
            auto sum = static_cast<T>(0);
            for (std::size_t k = 0; k < temp.size().second; ++k) {
                sum += temp[i][k] * matrix[k][j];
            }
            array[i][j] = sum;
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Scalar<T> U>
constexpr auto Matrix<T, I, J>::operator*=(const U& scalar) -> Matrix<T, I, J>&
{
    for (std::size_t i = 0; i < J; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            array[i][j] = array[i][j] * scalar;
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Scalar<T> U>
constexpr auto Matrix<T, I, J>::operator*=(const std::vector<U>& vector)
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
constexpr auto operator+(const Matrix<T, I, J>& lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<T, I, J>
{
    auto tmp = lhs;
    tmp += rhs;
    return tmp;
}

template <
    detail::Arithmetic T,
    detail::Arithmetic U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
constexpr auto operator-(const Matrix<T, I, J>& lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<T, I, J>
{
    auto tmp = lhs;
    tmp -= rhs;
    return tmp;
}

template <
    detail::Arithmetic T,
    detail::Arithmetic U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
constexpr auto operator*(const Matrix<T, I, J>& lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<T, I, B>
{
    auto tmp = lhs;

    return multiply(tmp, rhs);
}

template <
    detail::Arithmetic T,
    std::size_t I,
    std::size_t J,
    detail::Arithmetic U,
    std::size_t A,
    std::size_t B>
inline constexpr auto multiply(
    const Matrix<T, I, J>& lhs,
    const Matrix<U, A, B>& rhs) -> Matrix<T, I, B>
{
    static_assert(J == A, "Matrix::invalid size");

    Matrix<T, I, B> result;

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            T sum = 0;
            for (std::size_t k = 0; k < J; ++k) {
                sum += lhs[i][k] * rhs[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

template <
    detail::Arithmetic T,
    detail::Scalar<T> U,
    std::size_t I,
    std::size_t J>
constexpr auto operator*(const Matrix<T, I, J>& lhs, const U& rhs)
    -> Matrix<T, I, J>
{
    auto tmp = lhs;
    tmp *= rhs;
    return tmp;
}

template <
    detail::Arithmetic T,
    detail::Scalar<T> U,
    std::size_t I,
    std::size_t J>
constexpr auto operator*(U lhs, const Matrix<T, I, J>& rhs) -> Matrix<T, I, J>
{
    const_cast<Matrix<T, I, J>&>(rhs) *= lhs;
    return rhs;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator^(const unsigned int& power)
    -> Matrix<T, I, J>&
{
    this->power(power);

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr inline auto Matrix<T, I, J>::operator==(
    const Matrix<U, A, B>& matrix) const -> bool
{
    if constexpr (I != A or J != B or not detail::is_convertible_v<T, U>) {
        return false;
    }

    for (std::size_t i = 0; i < I; ++i) {
        for (std::size_t j = 0; j < J; ++j) {
            if (array[i][j] != matrix[i][j]) { return false; }
        }
    }

    return true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U>
constexpr inline auto Matrix<T, I, J>::operator==(
    const std::initializer_list<U>& list) const -> bool
{
    if (I * J != list.size() or not detail::is_convertible_v<T, U>) {
        return false;
    }

    auto row_num = 0;
    auto col_num = 0;
    for (const auto& elem : list) {
        if (array[row_num][col_num] != elem) { return false; }
        if (col_num != J - 1) { ++col_num; }
        else if (row_num != I - 1) {
            ++row_num;
            col_num = 0;
        }
    }

    return true;
}

// clang-format off
template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr inline auto Matrix<T, I, J>::operator!=(const Matrix<U, A, B>& matrix) const
    -> bool
{
    return not (*this == matrix);
}
// clang-format on

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator[](std::size_t row) -> Row<T, I, J>
{
    if (row > (I - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"
        };
    }
    return Row(*this, row);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator[](std::size_t row) const
    -> Crow<T, I, J>
{
    if (row > (I - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"
        };
    }
    return Crow(*this, row);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Row<T, I, J>::Row(Matrix<T, I, J>& matrix, std::size_t n_row)
    : matrix{ matrix }, n_row{ n_row }
{
    for (std::size_t i = 0; i < J; ++i) { row.emplace_back(matrix(n_row, i)); }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Row<T, I, J>::operator=(const std::vector<T>& new_row)
    -> Row<T, I, J>&
{
    if (not(row.size() == new_row.size())) {
        throw detail::exceptions::invalid_argument_input{ "Row::invalid size" };
    }

    row = new_row;

    std::size_t col_idx = 0;
    for (const auto elem : row) {
        matrix.underlying_array()[n_row][col_idx] = elem;
        ++col_idx;
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Row<T, I, J>::operator[](std::size_t col) -> T&
{
    if (col > (J - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"
        };
    }
    return matrix(n_row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Row<T, I, J>::get_row() const -> const std::vector<T>&
{
    return row;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Crow<T, I, J>::Crow(const Matrix<T, I, J>& matrix, std::size_t n_row)
    : matrix{ matrix }, n_row{ n_row }
{
    for (std::size_t i = 0; i < J; ++i) { row.emplace_back(matrix(n_row, i)); }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Crow<T, I, J>::operator[](std::size_t col) const -> T
{
    if (col > (J - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"
        };
    }
    return matrix(n_row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Crow<T, I, J>::get_row() const -> const std::vector<T>&
{
    return row;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col)
    -> T&
{
    if (row > (I - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"
        };
    }
    if (col > (J - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"
        };
    }

    return this->array[row][col];
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col)
    const -> const T&
{
    if (row > (I - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid row number"
        };
    }
    if (col > (J - 1)) {
        throw detail::exceptions::out_of_range_input{
            "Matrix::invalid col number"
        };
    }

    return this->array[row][col];
}

template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto operator<<(std::ostream& ostream, const Matrix<U, A, B>& array)
    -> std::ostream&
{
    for (std::size_t i = 0; i < array.size_.first; ++i) {
        for (std::size_t j = 0; j < array.size_.second; ++j) {
            ostream << array.array[i][j] << " ";
        }
        ostream << std::endl;
    }

    return ostream;
}

template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto operator<<(std::ostream& ostream, const Row<U, A, B>& row)
    -> std::ostream&
{
    for (const auto& elem : row.get_row()) {
        ostream << elem;
        ostream << " ";
    }

    return ostream;
}

template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto operator<<(std::ostream& ostream, const Crow<U, A, B>& row)
    -> std::ostream&
{
    for (const auto& elem : row.get_row()) {
        ostream << elem;
        ostream << " ";
    }

    return ostream;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::iterator::iterator(
    Matrix<T, I, J>& matrix_,
    std::size_t row_,
    std::size_t col_)
    : matrix{ matrix_ }, row{ row_ }, col{ col_ }
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::const_iterator::const_iterator(
    const Matrix<T, I, J>& matrix_,
    std::size_t row_,
    std::size_t col_)
    : matrix{ matrix_ }, row{ row_ }, col{ col_ }
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
constexpr auto Matrix<T, I, J>::begin() -> Matrix<T, I, J>::iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::begin() const -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::end() -> Matrix<T, I, J>::iterator
{
    return iterator(*this, I, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::end() const -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, I, 0);
}

}  // namespace mtl

#endif  // MTL_MATRIX_HPP