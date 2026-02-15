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

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef MTL_USE_GSL_SPAN
#include <gsl/span>
#else
#include <span>
#endif

namespace mtl {

#ifdef MTL_USE_GSL_SPAN
template <typename T, std::size_t Extent = gsl::dynamic_extent>
using Span = gsl::span<T, Extent>;
#else
template <typename T, std::size_t Extent = std::dynamic_extent>
using Span = std::span<T, Extent>;
#endif

template <typename T, std::size_t I, std::size_t J>
struct Row;

template <typename T, std::size_t I, std::size_t J>
struct Crow;

template <typename T, std::size_t I, std::size_t J>
struct Matrix final {
    static_assert(
        std::is_arithmetic_v<T>,
        "Matrix requires an arithmetic type");

   private:
    T _data[I][J]{};

   public:
    constexpr Matrix() noexcept;

    ~Matrix() noexcept = default;

    explicit constexpr Matrix(const T&) noexcept;

    template <typename U>
    explicit constexpr Matrix(const U&);

    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    constexpr Matrix(std::initializer_list<T>);

    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    constexpr Matrix(std::initializer_list<std::initializer_list<T>>);

    constexpr Matrix(const Matrix<T, I, J>&) noexcept;
    constexpr auto operator=(const Matrix<T, I, J>&) noexcept
        -> Matrix<T, I, J>&;

    constexpr Matrix(Matrix<T, I, J>&&) noexcept;
    constexpr auto operator=(Matrix<T, I, J>&&) noexcept -> Matrix<T, I, J>&;

    using array_type = T[I][J];
    using const_array_type = const T[I][J];

    [[nodiscard]] auto underlying_array() noexcept -> array_type&;
    [[nodiscard]] auto underlying_array() const noexcept -> const_array_type&;

    template <typename U, std::size_t A, std::size_t B>
    constexpr explicit Matrix(const Matrix<U, A, B>&);

    template <typename U, std::size_t A, std::size_t B>
    constexpr auto operator=(const Matrix<U, A, B>&) -> Matrix<T, I, J>&;

    template <typename U, std::size_t A, std::size_t B>
    constexpr Matrix(Matrix<U, A, B>&& matrix);

    template <typename U, std::size_t A, std::size_t B>
    constexpr auto operator=(Matrix<U, A, B>&& matrix) -> Matrix<T, I, J>&;

    template <typename U>
    constexpr auto operator=(const std::initializer_list<U>&)
        -> Matrix<T, I, J>&;

    template <typename U>
    constexpr auto operator=(std::initializer_list<U>&&) -> Matrix<T, I, J>&;

    template <typename U, std::size_t A, std::size_t B>
    explicit constexpr operator Matrix<U, A, B>() const;

    constexpr auto insert(const T&) noexcept -> void;

    [[nodiscard]] constexpr auto transpose() const noexcept -> Matrix<T, J, I>;

    [[nodiscard]] constexpr auto power(unsigned int) -> Matrix<T, I, J>;

    [[nodiscard]] constexpr auto det() const;

    [[nodiscard]] constexpr auto is_diagonal() const noexcept -> bool;

    [[nodiscard]] constexpr auto size() const noexcept
        -> std::pair<std::size_t, std::size_t>;

    [[nodiscard]] constexpr auto row_size() const noexcept -> std::size_t;

    [[nodiscard]] constexpr auto col_size() const noexcept -> std::size_t;

    constexpr auto clear() noexcept;

    template <typename U, std::size_t A, std::size_t B>
    constexpr auto operator+=(const Matrix<U, A, B>&) -> Matrix<T, I, J>&;

    template <typename U, std::size_t A, std::size_t B>
    constexpr auto operator-=(const Matrix<U, A, B>&) -> Matrix<T, I, J>&;

    template <typename U, std::size_t A, std::size_t B>
    constexpr auto operator*=(const Matrix<U, A, B>&) -> Matrix<T, I, B>&;

    template <typename U>
    constexpr auto operator*=(const U&) -> Matrix<T, I, J>&;

    template <typename U>
    constexpr auto operator*=(const std::vector<U>&) -> Matrix<T, I, J>&;

    constexpr auto operator^(const unsigned int&) -> Matrix<T, I, J>&;

    template <typename U, std::size_t A, std::size_t B>
    [[nodiscard]] constexpr inline auto operator==(
        const Matrix<U, A, B>&) const noexcept -> bool;

    template <typename U>
    [[nodiscard]] constexpr inline auto operator==(
        const std::initializer_list<U>&) const noexcept -> bool;

    template <typename U, std::size_t A, std::size_t B>
    [[nodiscard]] constexpr inline auto operator!=(
        const Matrix<U, A, B>&) const noexcept -> bool;

    constexpr auto operator[](std::size_t) -> Row<T, I, J>;

    constexpr auto operator[](std::size_t) const -> Crow<T, I, J>;

    constexpr auto operator()(std::size_t, std::size_t) -> T&;
    constexpr auto operator()(std::size_t, std::size_t) const -> const T&;

#ifdef MTL_ENABLE_OSTREAM
    template <typename U, std::size_t A, std::size_t B>
    friend constexpr auto operator<<(std::ostream&, const Matrix<U, A, B>&)
        -> std::ostream&;
#endif

    struct iterator {
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
        using const_reference = const T&;

        iterator(Matrix<T, I, J>&, std::size_t, std::size_t) noexcept;
        auto operator*() noexcept -> reference;
        auto operator*() const noexcept -> const_reference;
        auto operator++() noexcept -> iterator&;
        auto operator++(int) noexcept -> iterator;
        inline auto operator==(const iterator&) const noexcept -> bool;
        inline auto operator!=(const iterator&) const noexcept -> bool;
        friend struct Matrix;
    };

    struct const_iterator {
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
        using const_reference = const T&;

        const_iterator(
            const Matrix<T, I, J>&,
            std::size_t,
            std::size_t) noexcept;
        auto operator*() const noexcept -> const_reference;
        auto operator++() noexcept -> const_iterator&;
        auto operator++(int) noexcept -> const_iterator;
        inline auto operator==(const const_iterator&) const noexcept -> bool;
        inline auto operator!=(const const_iterator&) const noexcept -> bool;
        friend struct Matrix;
    };

    auto begin() noexcept -> iterator;
    auto end() noexcept -> iterator;

    auto begin() const noexcept -> const_iterator;
    auto end() const noexcept -> const_iterator;
};

template <typename T, std::size_t I, std::size_t J>
struct Row {
   private:
    Matrix<T, I, J>& _matrix;
    std::size_t _rowNumber;

   public:
    explicit Row(Matrix<T, I, J>& mat, std::size_t row_idx)
        : _matrix{ mat }, _rowNumber{ row_idx }
    {
    }

    auto operator[](std::size_t col) -> T& { return _matrix(_rowNumber, col); }

    auto get_row() -> Span<T, J>
    {
        return Span<T, J>{ _matrix.underlying_array()[_rowNumber], J };
    }

    auto get_row() const -> Span<const T, J>
    {
        return Span<const T, J>{ _matrix.underlying_array()[_rowNumber], J };
    }
};

template <typename T, std::size_t I, std::size_t J>
struct Crow {
   private:
    const Matrix<T, I, J>& _matrix;
    std::size_t _rowNumber;

   public:
    explicit Crow(const Matrix<T, I, J>& mat, std::size_t row_idx)
        : _matrix{ mat }, _rowNumber{ row_idx }
    {
    }

    auto operator[](std::size_t col) const -> T
    {
        return _matrix(_rowNumber, col);
    }

    auto get_row() const -> Span<const T, J>
    {
        return Span<const T, J>{ _matrix.underlying_array()[_rowNumber], J };
    }
};

template <typename T>
Matrix(T) -> Matrix<T, 1, 1>;

template <typename T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix() noexcept : _data{}
{
}

template <typename T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(const T& value) noexcept
{
    std::fill(begin(), end(), value);
}

template <typename T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(std::initializer_list<T> elems)
{
    std::size_t row_num = 0;
    std::size_t col_num = 0;
    for (const auto& elem : elems) {
        _data[row_num][col_num] = elem;
        if (col_num != col_size() - 1) { ++col_num; }
        else if (row_num != row_size() - 1) {
            ++row_num;
            col_num = 0;
        }
    }
}

template <typename T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(
    std::initializer_list<std::initializer_list<T>> elems)
{
    for (const auto& elem : elems) {
        if (elems.size() * elem.size() != row_size() * col_size()) { return; }
    }

    std::size_t i = 0;
    std::size_t j = 0;
    for (const auto& row : elems) {
        for (const auto& item : row) {
            _data[i][j] = item;
            if (j != col_size() - 1) { ++j; }
            else if (i != row_size() - 1) {
                ++i;
                j = 0;
            }
        }
    }
}

template <typename T, std::size_t I, std::size_t J>
template <typename U>
constexpr Matrix<T, I, J>::Matrix(const U& value)
{
    static_assert(std::is_convertible_v<T, U>, "Matrix::Matrix() invalid type");

    for (auto i = 0; i < row_size(); ++i) {
        for (auto j = 0; j < col_size(); ++j) {
            _data[i][j] = static_cast<T>(value);
        }
    }
}

template <typename T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(const Matrix<T, I, J>& matrix) noexcept =
    default;

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(const Matrix<T, I, J>& other) noexcept
    -> Matrix<T, I, J>&
{
    if (this != &other) {
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(_data, other._data, sizeof(_data));
        }
        else {
            std::copy(
                &other._data[0][0],
                &other._data[0][0] + I * J,
                &_data[0][0]);
        }
    }
    return *this;
}

template <typename T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(Matrix<T, I, J>&& matrix) noexcept = default;

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(Matrix<T, I, J>&& other) noexcept
    -> Matrix<T, I, J>&
{
    if (this != &other) {
        using std::make_move_iterator;
        std::copy(
            make_move_iterator(&other._data[0][0]),
            make_move_iterator(&other._data[0][0] + I * J),
            &_data[0][0]);
    }
    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr Matrix<T, I, J>::Matrix(const Matrix<U, A, B>& matrix)
{
    static_assert(std::is_convertible_v<T, U>, "Matrix::Matrix() invalid type");
    static_assert(I == A and J == B, "Matrix::Matrix() invalid size");

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            _data[i][j] = static_cast<T>(matrix(i, j));
        }
    }
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(
        std::is_convertible_v<T, U>,
        "Matrix::operator= invalid type");
    static_assert(I == A and J == B, "Matrix::operator= invalid size");

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            _data[i][j] = static_cast<T>(matrix(i, j));
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr Matrix<T, I, J>::Matrix(Matrix<U, A, B>&& matrix)
{
    static_assert(std::is_convertible_v<T, U>, "Matrix::Matrix() invalid type");
    static_assert(I == A and J == B, "Matrix::Matrix() invalid size");

    *this = std::move(matrix);
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(Matrix<U, A, B>&& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(
        std::is_convertible_v<T, U>,
        "Matrix::operator= invalid type");
    static_assert(I == A and J == B, "Matrix::operator= invalid size");

    *this = std::move(matrix);

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U>
constexpr auto Matrix<T, I, J>::operator=(const std::initializer_list<U>& list)
    -> Matrix<T, I, J>&
{
    static_assert(
        std::is_convertible_v<T, U>,
        "Matrix::operator= invalid type");

    std::size_t row_num = 0;
    std::size_t col_num = 0;
    for (const auto& elem : list) {
        _data[row_num][col_num] = elem;
        if (col_num != col_size() - 1) { ++col_num; }
        else if (row_num != row_size() - 1) {
            ++row_num;
            col_num = 0;
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U>
constexpr auto Matrix<T, I, J>::operator=(std::initializer_list<U>&& list)
    -> Matrix<T, I, J>&
{
    static_assert(
        std::is_convertible_v<T, U>,
        "Matrix::operator= invalid type");

    std::size_t row_num = 0;
    std::size_t col_num = 0;
    for (const auto& elem : list) {
        _data[row_num][col_num] = std::move(elem);
        if (col_num != col_size() - 1) { ++col_num; }
        else if (row_num != row_size() - 1) {
            ++row_num;
            col_num = 0;
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
inline constexpr Matrix<T, I, J>::operator Matrix<U, A, B>() const
{
    static_assert(
        std::is_convertible_v<T, U>,
        "Matrix::operator Matrix<U, A, B>() invalid type");
    static_assert(
        I <= A and J <= B,
        "Matrix::operator Matrix<U, A, B>() invalid size");

    Matrix<U, A, B> result;
    for (std::size_t i = 0; i < A; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            if (i < I and j < J) { result[i][j] = static_cast<U>(_data[i][j]); }
            else {
                result[i][j] = U();
            }
        }
    }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::underlying_array() noexcept -> array_type&
{
    return _data;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::underlying_array() const noexcept -> const_array_type&
{
    return _data;
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::insert(const T& element) noexcept -> void
{
    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) { _data[i][j] = element; }
    }
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::transpose() const noexcept -> Matrix<T, J, I>
{
    Matrix<T, J, I> result{};

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            result.underlying_array()[j][i] = _data[i][j];
        }
    }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::det() const
{
    static_assert(I == J, "Matrix::det() is only defined for square matrices");

    double determinant = 1;
    Matrix<double, I, J> temp(*this);

    for (std::size_t i = 0; i < row_size(); ++i) {
        std::size_t non_zero_row = i;
        while (non_zero_row < row_size() && temp[non_zero_row][i] == 0) {
            ++non_zero_row;
        }

        if (non_zero_row == row_size()) { return 0.0; }

        if (non_zero_row != i) {
            for (std::size_t j = 0; j < col_size(); ++j) {
                std::swap(temp[i][j], temp[non_zero_row][j]);
            }
            determinant *= -1;
        }

        double pivot = temp[i][i];
        determinant *= pivot;

        for (std::size_t j = 0; j < col_size(); ++j) { temp[i][j] /= pivot; }

        for (std::size_t k = i + 1; k < row_size(); ++k) {
            double factor = temp[k][i];
            for (std::size_t j = 0; j < col_size(); ++j) {
                temp[k][j] -= factor * temp[i][j];
            }
        }
    }

    constexpr auto roundhelper = [](double value, int precision) {
        constexpr double base = 10.0;
        double multiplier = std::pow(base, precision);
        return std::round(value * multiplier) / multiplier;
    };

    constexpr int precision = 5;
    return roundhelper(determinant, precision);
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::is_diagonal() const noexcept -> bool
{
    if (row_size() != col_size()) { return false; }

    for (std::size_t row_num = 0; row_num < row_size(); ++row_num) {
        for (std::size_t col_num = 0; col_num < col_size(); ++col_num) {
            if (row_num != col_num and _data[row_num][col_num] != 0) {
                return false;
            }
        }
    }

    return true;
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size() const noexcept
    -> std::pair<std::size_t, std::size_t>
{
    return std::make_pair(I, J);
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::row_size() const noexcept -> std::size_t
{
    return size().first;
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::col_size() const noexcept -> std::size_t
{
    return size().second;
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::clear() noexcept
{
    std::fill(begin(), end(), 0);
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::power(unsigned int power) -> Matrix<T, I, J>
{
    auto result = *this;
    for (unsigned int i = 0; i < (power - 1); ++i) { result *= *this; }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator+=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(
        std::is_convertible_v<U, T>,
        "Matrix::operator+=: invalid type");
    static_assert(I == A and J == B, "Matrix::operator+=: invalid size");

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            _data[i][j] += static_cast<T>(matrix.underlying_array()[i][j]);
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator-=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    static_assert(
        std::is_convertible_v<U, T>,
        "Matrix::operator-=: invalid type");
    static_assert(I == A and J == B, "Matrix::operator-=: invalid size");

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            _data[i][j] -= static_cast<T>(matrix.underlying_array()[i][j]);
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator*=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, B>&
{
    auto temp = *this;

    static_assert(
        std::is_convertible_v<U, T>,
        "Matrix::operator*=: invalid type");
    static_assert(J == A, "Matrix::operator*=: invalid size");

    for (std::size_t i = 0; i < temp.row_size(); ++i) {
        for (std::size_t j = 0; j < matrix.col_size(); ++j) {
            auto sum = static_cast<T>(0);
            for (std::size_t k = 0; k < temp.col_size(); ++k) {
                sum += temp[i][k] * matrix[k][j];
            }
            _data[i][j] = sum;
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U>
constexpr auto Matrix<T, I, J>::operator*=(const U& scalar) -> Matrix<T, I, J>&
{
    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            _data[i][j] = _data[i][j] * static_cast<T>(scalar);
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U>
constexpr auto Matrix<T, I, J>::operator*=(const std::vector<U>& vector)
    -> Matrix<T, I, J>&
{
    static_assert(
        std::is_convertible_v<U, T>,
        "Matrix::operator*=: invalid type");
    static_assert(
        J == vector.size(),
        "Matrix::operator*=: invalid vector size");

    const auto temp = *this;

    for (std::size_t i = 0; i < row_size(); ++i) {
        _data[i][0] = 0;
        for (std::size_t j = 0; j < col_size(); ++j) {
            _data[i][0] += temp[i][j] * vector[j];
        }
    }

    return *this;
}

template <
    typename T,
    typename U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
constexpr auto operator+(const Matrix<T, I, J>& lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<std::common_type_t<T, U>, I, J>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);
    tmp += rhs;
    return tmp;
}

template <
    typename T,
    typename U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
constexpr auto operator-(const Matrix<T, I, J>& lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<std::common_type_t<T, U>, I, J>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);
    tmp -= rhs;
    return tmp;
}

template <
    typename T,
    typename U,
    std::size_t I,
    std::size_t J,
    std::size_t A,
    std::size_t B>
constexpr auto operator*(const Matrix<T, I, J>& lhs, const Matrix<U, A, B>& rhs)
    -> Matrix<std::common_type_t<T, U>, I, B>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);

    return multiply(tmp, rhs);
}

template <
    typename T,
    std::size_t I,
    std::size_t J,
    typename U,
    std::size_t A,
    std::size_t B>
inline constexpr auto multiply(
    const Matrix<T, I, J>& lhs,
    const Matrix<U, A, B>& rhs) -> Matrix<std::common_type_t<T, U>, I, B>
{
    static_assert(
        std::is_convertible_v<U, T>,
        "Matrix::multiply: invalid type");
    static_assert(J == A, "Matrix::multiply: invalid size");

    Matrix<std::common_type_t<T, U>, I, B> result;

    for (std::size_t i = 0; i < lhs.row_size(); ++i) {
        for (std::size_t j = 0; j < rhs.col_size(); ++j) {
            T sum = 0;
            for (std::size_t k = 0; k < lhs.col_size(); ++k) {
                sum += lhs[i][k] * rhs[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

template <typename T, typename U, std::size_t I, std::size_t J>
constexpr auto operator*(const Matrix<T, I, J>& lhs, const U& rhs)
    -> Matrix<std::common_type_t<T, U>, I, J>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);
    tmp *= rhs;
    return tmp;
}

template <typename T, typename U, std::size_t I, std::size_t J>
constexpr auto operator*(U lhs, const Matrix<T, I, J>& rhs) -> Matrix<T, I, J>
{
    const_cast<Matrix<T, I, J>&>(rhs) *= lhs;
    return rhs;
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator^(const unsigned int& power)
    -> Matrix<T, I, J>&
{
    *this = this->power(power);

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr inline auto Matrix<T, I, J>::operator==(
    const Matrix<U, A, B>& matrix) const noexcept -> bool
{
    if (row_size() != matrix.row_size() or col_size() != matrix.col_size()
        or not std::is_convertible_v<T, U>) {
        return false;
    }

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            if (_data[i][j] != matrix.underlying_array()[i][j]) {
                return false;
            }
        }
    }

    return true;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U>
constexpr inline auto Matrix<T, I, J>::operator==(
    const std::initializer_list<U>& list) const noexcept -> bool
{
    if (row_size() * col_size() != list.size()
        or not std::is_convertible_v<T, U>) {
        return false;
    }

    std::size_t row_num = 0;
    std::size_t col_num = 0;
    for (const auto& elem : list) {
        if (_data[row_num][col_num] != elem) { return false; }
        if (col_num != col_size() - 1) { ++col_num; }
        else if (row_num != row_size() - 1) {
            ++row_num;
            col_num = 0;
        }
    }

    return true;
}

// clang-format off
template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
constexpr inline auto Matrix<T, I, J>::operator!=(const Matrix<U, A, B>& matrix) const noexcept
    -> bool
{
    return not (*this == matrix);
}
// clang-format on

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator[](std::size_t row) -> Row<T, I, J>
{
    return Row(*this, row);
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator[](std::size_t row) const
    -> Crow<T, I, J>
{
    return Crow(*this, row);
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col)
    -> T&
{
    return _data[row][col];
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col)
    const -> const T&
{
    return _data[row][col];
}

#ifdef MTL_ENABLE_OSTREAM
template <typename U, std::size_t A, std::size_t B>
constexpr auto operator<<(std::ostream& ostream, const Matrix<U, A, B>& matrix)
    -> std::ostream&
{
    for (std::size_t i = 0; i < matrix.size_.first; ++i) {
        for (std::size_t j = 0; j < matrix.size_.second; ++j) {
            ostream << matrix._data[i][j] << " ";
        }
        ostream << "\n";
    }

    return ostream;
}

template <typename U, std::size_t A, std::size_t B>
constexpr auto operator<<(std::ostream& ostream, const Row<U, A, B>& row)
    -> std::ostream&
{
    for (const auto& elem : row.get_row()) {
        ostream << elem;
        ostream << " ";
    }

    return ostream;
}

template <typename U, std::size_t A, std::size_t B>
constexpr auto operator<<(std::ostream& ostream, const Crow<U, A, B>& row)
    -> std::ostream&
{
    for (const auto& elem : row.get_row()) {
        ostream << elem;
        ostream << " ";
    }

    return ostream;
}
#endif

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::iterator::iterator(
    Matrix<T, I, J>& matrix_,
    std::size_t row_,
    std::size_t col_) noexcept
    : matrix{ matrix_ }, row{ row_ }, col{ col_ }
{
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::const_iterator::const_iterator(
    const Matrix<T, I, J>& matrix_,
    std::size_t row_,
    std::size_t col_) noexcept
    : matrix{ matrix_ }, row{ row_ }, col{ col_ }
{
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator*() noexcept -> T&
{
    return matrix.underlying_array()[row][col];
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator*() const noexcept -> const T&
{
    return matrix.underlying_array()[row][col];
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator*() const noexcept -> const T&
{
    if (row >= matrix.row_size() or col >= matrix.col_size()) {
        static T default_value{};
        return default_value;
    }
    return matrix(row, col);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++() noexcept
    -> Matrix<T, I, J>::iterator&
{
    ++col;
    if (col == matrix.col_size()) {
        col = 0;
        ++row;
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator++() noexcept
    -> Matrix<T, I, J>::const_iterator&
{
    ++col;
    if (col == matrix.col_size()) {
        col = 0;
        ++row;
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++(int) noexcept
    -> Matrix<T, I, J>::iterator
{
    auto temp = *this;

    ++(*this);

    return temp;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator++(int) noexcept
    -> Matrix<T, I, J>::const_iterator
{
    auto temp = *this;

    ++(*this);

    return temp;
}

template <typename T, std::size_t I, std::size_t J>
inline auto Matrix<T, I, J>::iterator::operator==(
    const iterator& iter) const noexcept -> bool
{
    return row == iter.row and col == iter.col;
}

template <typename T, std::size_t I, std::size_t J>
inline auto Matrix<T, I, J>::const_iterator::operator==(
    const const_iterator& iter) const noexcept -> bool
{
    return row == iter.row and col == iter.col;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator!=(const iterator& iter) const noexcept
    -> bool
{
    return not(*this == iter);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator!=(
    const const_iterator& iter) const noexcept -> bool
{
    return not(*this == iter);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::begin() noexcept -> Matrix<T, I, J>::iterator
{
    return iterator(*this, 0, 0);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::begin() const noexcept -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, 0, 0);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::end() noexcept -> Matrix<T, I, J>::iterator
{
    return iterator(*this, row_size(), 0);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::end() const noexcept -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, row_size(), 0);
}

}  // namespace mtl

#endif  // MTL_MATRIX_HPP