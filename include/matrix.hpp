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
struct Row;

template <detail::Arithmetic T, std::size_t I, std::size_t J>
struct Crow;

// NOLINTBEGIN(hicpp-named-parameter,readability-named-parameter)
template <detail::Arithmetic T, std::size_t I, std::size_t J>
struct Matrix final {
   private:
    T** data{ nullptr };
    std::pair<std::size_t, std::size_t> size_{ I, J };
    bool has_been_reallocated{ false };

   public:
    constexpr Matrix() noexcept;

    constexpr ~Matrix() noexcept;

    explicit constexpr Matrix(const T&) noexcept;

    template <detail::Arithmetic U>
    explicit constexpr Matrix(const U&);

    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    constexpr Matrix(std::initializer_list<T>);

    // NOLINTNEXTLINE(hicpp-explicit-conversions)
    constexpr Matrix(std::initializer_list<std::initializer_list<T>>);

    constexpr Matrix(const Matrix<T, I, J>&) noexcept;
    constexpr auto operator=(const Matrix<T, I, J>&) noexcept
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr explicit Matrix(const Matrix<U, A, B>&);

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator=(const Matrix<U, A, B>&) -> Matrix<T, I, J>&;

    constexpr Matrix(Matrix<T, I, J>&&) noexcept;
    constexpr auto operator=(Matrix<T, I, J>&&) noexcept -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr explicit Matrix(Matrix<U, A, B>&&);

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator=(Matrix<U, A, B>&&) -> Matrix<T, I, J>&;

    template <detail::Arithmetic U>
    constexpr auto operator=(const std::initializer_list<U>&)
        -> Matrix<T, I, J>&;

    template <detail::Arithmetic U>
    constexpr auto operator=(std::initializer_list<U>&&) -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    explicit constexpr operator Matrix<U, A, B>() const;

    [[nodiscard]] constexpr auto underlying_array() const noexcept -> T**;

    constexpr auto insert(const T&) noexcept;

    constexpr auto sort();

    [[nodiscard]] constexpr auto transpose() const noexcept -> Matrix<T, J, I>;

    [[nodiscard]] constexpr auto power(unsigned int) -> Matrix<T, I, J>;

    [[nodiscard]] constexpr auto det() const;

    [[nodiscard]] constexpr auto is_diagonal() const noexcept -> bool;

   private:
    constexpr auto zeros() noexcept;

   public:
    constexpr auto alloc() noexcept;
    constexpr auto alloc() const noexcept;

   private:
    constexpr auto alloc(std::size_t, std::size_t) noexcept;
    constexpr auto alloc(std::size_t, std::size_t) const noexcept;

   public:
    constexpr auto realloc(std::size_t, std::size_t) noexcept;

    constexpr auto dealloc() noexcept;
    constexpr auto dealloc() const noexcept;

    [[nodiscard]] constexpr auto size() const noexcept
        -> std::pair<std::size_t, std::size_t>;

    [[nodiscard]] constexpr auto row_size() const noexcept -> std::size_t;

    [[nodiscard]] constexpr auto col_size() const noexcept -> std::size_t;

    [[nodiscard]] constexpr auto is_reallocated() const noexcept -> bool;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator+=(const Matrix<U, A, B>&) -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator-=(const Matrix<U, A, B>&) -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    constexpr auto operator*=(const Matrix<U, A, B>&) -> Matrix<T, I, B>&;

    template <detail::Scalar<T> U>
    constexpr auto operator*=(const U&) -> Matrix<T, I, J>&;

    template <detail::Scalar<T> U>
    constexpr auto operator*=(const std::vector<U>&) -> Matrix<T, I, J>&;

    constexpr auto operator^(const unsigned int&) -> Matrix<T, I, J>&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    [[nodiscard]] constexpr inline auto operator==(
        const Matrix<U, A, B>&) const noexcept -> bool;

    template <detail::Arithmetic U>
    [[nodiscard]] constexpr inline auto operator==(
        const std::initializer_list<U>&) const noexcept -> bool;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    [[nodiscard]] constexpr inline auto operator!=(
        const Matrix<U, A, B>&) const noexcept -> bool;

    constexpr auto operator[](std::size_t) -> Row<T, I, J>;

    constexpr auto operator[](std::size_t) const -> Crow<T, I, J>;

    constexpr auto operator()(std::size_t, std::size_t) const -> T&;

    constexpr auto at(std::size_t, std::size_t) const -> T&;

    template <detail::Arithmetic U, std::size_t A, std::size_t B>
    friend constexpr auto operator<<(std::ostream&, const Matrix<U, A, B>&)
        -> std::ostream&;

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

template <detail::Arithmetic T, std::size_t I, std::size_t J>
struct Row {
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
struct Crow {
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

template <detail::Arithmetic T>
Matrix(T) -> Matrix<T, 1, 1>;

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix() noexcept
{
    alloc();

    zeros();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::~Matrix() noexcept
{
    dealloc();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(const T& value) noexcept
{
    alloc();

    std::fill(begin(), end(), value);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(std::initializer_list<T> elems)
{
    alloc();

    if (elems.size() != (row_size() * col_size())) [[unlikely]] {
        throw std::invalid_argument{
            "Matrix::Matrix() cannot initialize matrix with incorrect "
            "number of elements."
        };
    }
    else [[likely]] {
        std::size_t row_num = 0;
        std::size_t col_num = 0;
        for (const auto& elem : elems) {
            data[row_num][col_num] = elem;
            if (col_num != col_size() - 1) { ++col_num; }
            else if (row_num != row_size() - 1) {
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

    for (const auto& elem : elems) {
        if (elems.size() * elem.size() != row_size() * col_size())
            [[unlikely]] {
            throw std::invalid_argument{
                "Matrix::Matrix() cannot initialize matrix with incorrect "
                "number of elements."
            };
            return;
        }
    }

    std::size_t i = 0;
    std::size_t j = 0;
    for (const auto& row : elems) {
        for (const auto& item : row) {
            data[i][j] = item;
            if (j != col_size() - 1) { ++j; }
            else if (i != row_size() - 1) {
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

    if constexpr (not std::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::Matrix(), inconvertible types" };
    }

    for (auto i = 0; i < row_size(); ++i) {
        for (auto j = 0; j < col_size(); ++j) {
            data[i][j] = static_cast<T>(value);
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(const Matrix<T, I, J>& matrix) noexcept
    : size_{ matrix.size() }, has_been_reallocated{ matrix.is_reallocated() }
{
    alloc();

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            data[i][j] = matrix.data[i][j];
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(
    const Matrix<T, I, J>& matrix) noexcept -> Matrix<T, I, J>&
{
    size_ = matrix.size();
    has_been_reallocated = matrix.is_reallocated();

    alloc();

    if (&matrix != this) {
        for (std::size_t i = 0; i < row_size(); ++i) {
            for (std::size_t j = 0; j < col_size(); ++j) {
                data[i][j] = matrix.data[i][j];
            }
        }
    }
    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr Matrix<T, I, J>::Matrix(Matrix<T, I, J>&& matrix) noexcept
{
    *this = std::move(matrix);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(Matrix<T, I, J>&& matrix) noexcept
    -> Matrix<T, I, J>&
{
    if (&matrix != this) {
        data = std::exchange(matrix.data, nullptr);
        size_ = std::exchange(matrix.size_, { 0, 0 });
        has_been_reallocated =
            std::exchange(matrix.has_been_reallocated, false);
    }
    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr Matrix<T, I, J>::Matrix(const Matrix<U, A, B>& matrix)
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (size() != matrix.size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    size_ = matrix.size();
    has_been_reallocated = matrix.is_reallocated();

    alloc();

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            data[i][j] = static_cast<T>(matrix.underlying_array()[i][j]);
        }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (size() != matrix.size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    size_ = matrix.size();
    has_been_reallocated = matrix.is_reallocated();

    alloc();

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            data[i][j] = static_cast<T>(matrix.underlying_array()[i][j]);
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr Matrix<T, I, J>::Matrix(Matrix<U, A, B>&& matrix) : size_{ 0, 0 }
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (size() != matrix.size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    *this = std::move(matrix);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator=(Matrix<U, A, B>&& matrix)
    -> Matrix<T, I, J>&
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }
    else if (size() != matrix.size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }
    else {
        *this = std::move(matrix);

        return *this;
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U>
constexpr auto Matrix<T, I, J>::operator=(const std::initializer_list<U>& list)
    -> Matrix<T, I, J>&
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (row_size() * col_size() != list.size()) {
        throw std::invalid_argument{
            "Matrix::operator= std::initializer_list invalid size"
        };
    }

    std::size_t row_num = 0;
    std::size_t col_num = 0;
    for (const auto& elem : list) {
        data[row_num][col_num] = elem;
        if (col_num != col_size() - 1) { ++col_num; }
        else if (row_num != row_size() - 1) {
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
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (row_size() * col_size() != list.size()) {
        throw std::invalid_argument{
            "Matrix::operator= std::initializer_list invalid size"
        };
    }

    std::size_t row_num = 0;
    std::size_t col_num = 0;
    for (const auto& elem : list) {
        data[row_num][col_num] = std::move(elem);
        if (col_num != col_size() - 1) { ++col_num; }
        else if (row_num != row_size() - 1) {
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
    if constexpr (A < I or B < J) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    Matrix<U, A, B> result;
    for (std::size_t i = 0; i < A; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            if (i < I and j < J) { result[i][j] = static_cast<U>(data[i][j]); }
            else {
                result[i][j] = U();
            }
        }
    }

    return result;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::underlying_array() const noexcept -> T**
{
    return data;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::insert(const T& element) noexcept
{
    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) { data[i][j] = element; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::sort()
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::transpose() const noexcept -> Matrix<T, J, I>
{
    Matrix<T, J, I> result{};

    if (has_been_reallocated) { result.realloc(col_size(), row_size()); }

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            result.underlying_array()[j][i] = this->underlying_array()[i][j];
        }
    }

    return result;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::det() const
{
    if (row_size() != col_size()) {
        throw std::logic_error{ "Matrix::det: invalid size" };
    }

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

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::is_diagonal() const noexcept -> bool
{
    if (row_size() != col_size()) { return false; }

    for (std::size_t row_num = 0; row_num < row_size(); ++row_num) {
        for (std::size_t col_num = 0; col_num < col_size(); ++col_num) {
            if (row_num != col_num and data[row_num][col_num] != 0) {
                return false;
            }
        }
    }

    return true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::zeros() noexcept
{
    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) { data[i][j] = 1; }
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc() noexcept
{
    data = new T*[row_size()];
    for (std::size_t i = 0; i < row_size(); ++i) {
        data[i] = new T[col_size()];
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc() const noexcept
{
    data = new T*[row_size()];
    for (std::size_t i = 0; i < row_size(); ++i) {
        data[i] = new T[col_size()];
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc(
    std::size_t row_size_,
    std::size_t col_size_) noexcept
{
    data = new T*[row_size_];
    for (std::size_t i = 0; i < row_size_; ++i) { data[i] = new T[col_size_]; }

    size_ = std::make_pair(row_size_, col_size_);

    zeros();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::alloc(
    std::size_t row_size_,
    std::size_t col_size_) const noexcept
{
    data = new T*[row_size_];
    for (std::size_t i = 0; i < row_size_; ++i) { data[i] = new T[col_size_]; }

    zeros();
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::realloc(
    std::size_t row_size_,
    std::size_t col_size_) noexcept
{
    dealloc();

    size_ = std::make_pair(row_size_, col_size_);

    alloc(row_size_, col_size_);

    has_been_reallocated = true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::dealloc() noexcept
{
    for (std::size_t i = 0; i < row_size(); ++i) { delete[] data[i]; }
    delete[] data;
    data = nullptr;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::dealloc() const noexcept
{
    for (std::size_t i = 0; i < row_size(); ++i) { delete[] data[i]; }
    delete[] data;
    data = nullptr;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size() const noexcept
    -> std::pair<std::size_t, std::size_t>
{
    return size_;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::row_size() const noexcept -> std::size_t
{
    return size_.first;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::col_size() const noexcept -> std::size_t
{
    return size_.second;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::is_reallocated() const noexcept -> bool
{
    return has_been_reallocated;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::power(unsigned int power) -> Matrix<T, I, J>
{
    auto result = *this;
    for (unsigned int i = 0; i < (power - 1); ++i) { result *= *this; }

    return result;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator+=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (size() != matrix.size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            data[i][j] += static_cast<T>(matrix.underlying_array()[i][j]);
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto Matrix<T, I, J>::operator-=(const Matrix<U, A, B>& matrix)
    -> Matrix<T, I, J>&
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (size() != matrix.size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            data[i][j] -= static_cast<T>(matrix.underlying_array()[i][j]);
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

    if (col_size() != matrix.row_size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    for (std::size_t i = 0; i < temp.row_size(); ++i) {
        for (std::size_t j = 0; j < matrix.col_size(); ++j) {
            auto sum = static_cast<T>(0);
            for (std::size_t k = 0; k < temp.col_size(); ++k) {
                sum += temp[i][k] * matrix[k][j];
            }
            data[i][j] = sum;
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Scalar<T> U>
constexpr auto Matrix<T, I, J>::operator*=(const U& scalar) -> Matrix<T, I, J>&
{
    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            data[i][j] = data[i][j] * static_cast<T>(scalar);
        }
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Scalar<T> U>
constexpr auto Matrix<T, I, J>::operator*=(const std::vector<U>& vector)
    -> Matrix<T, I, J>&
{
    if constexpr (not detail::is_convertible_v<T, U>) {
        throw std::logic_error{ "Matrix::invalid type" };
    }

    if (col_size() != vector.size()) {
        throw std::logic_error{ "Matrix::invalid vector size" };
    }

    const auto temp = *this;

    for (std::size_t i = 0; i < row_size(); ++i) {
        data[i][0] = 0;
        for (std::size_t j = 0; j < col_size(); ++j) {
            data[i][0] += temp[i][j] * vector[j];
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
    -> Matrix<std::common_type_t<T, U>, I, J>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);
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
    -> Matrix<std::common_type_t<T, U>, I, J>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);
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
    -> Matrix<std::common_type_t<T, U>, I, B>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);

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
    const Matrix<U, A, B>& rhs) -> Matrix<std::common_type_t<T, U>, I, B>
{
    if (lhs.col_size() != rhs.row_size()) {
        throw std::logic_error{ "Matrix::invalid size" };
    }

    Matrix<std::common_type_t<T, U>, I, B> result;

    if (lhs.is_reallocated() or rhs.is_reallocated()) {
        result.realloc(lhs.row_size(), rhs.col_size());
    }

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

template <
    detail::Arithmetic T,
    detail::Scalar<T> U,
    std::size_t I,
    std::size_t J>
constexpr auto operator*(const Matrix<T, I, J>& lhs, const U& rhs)
    -> Matrix<std::common_type_t<T, U>, I, J>
{
    auto tmp = static_cast<Matrix<std::common_type_t<T, U>, I, J>>(lhs);
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
    *this = this->power(power);

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr inline auto Matrix<T, I, J>::operator==(
    const Matrix<U, A, B>& matrix) const noexcept -> bool
{
    if (row_size() != matrix.row_size() or col_size() != matrix.col_size()
        or not detail::is_convertible_v<T, U>) {
        return false;
    }

    for (std::size_t i = 0; i < row_size(); ++i) {
        for (std::size_t j = 0; j < col_size(); ++j) {
            if (data[i][j] != matrix.underlying_array()[i][j]) { return false; }
        }
    }

    return true;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U>
constexpr inline auto Matrix<T, I, J>::operator==(
    const std::initializer_list<U>& list) const noexcept -> bool
{
    if (row_size() * col_size() != list.size()
        or not detail::is_convertible_v<T, U>) {
        return false;
    }

    std::size_t row_num = 0;
    std::size_t col_num = 0;
    for (const auto& elem : list) {
        if (data[row_num][col_num] != elem) { return false; }
        if (col_num != col_size() - 1) { ++col_num; }
        else if (row_num != row_size() - 1) {
            ++row_num;
            col_num = 0;
        }
    }

    return true;
}

// clang-format off
template <detail::Arithmetic T, std::size_t I, std::size_t J>
template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr inline auto Matrix<T, I, J>::operator!=(const Matrix<U, A, B>& matrix) const noexcept
    -> bool
{
    return not (*this == matrix);
}
// clang-format on

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator[](std::size_t row) -> Row<T, I, J>
{
    if (row > (row_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid row number" };
    }
    return Row(*this, row);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator[](std::size_t row) const
    -> Crow<T, I, J>
{
    if (row > (row_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid row number" };
    }
    return Crow(*this, row);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Row<T, I, J>::Row(Matrix<T, I, J>& matrix, std::size_t n_row)
    : matrix{ matrix }, n_row{ n_row }
{
    for (std::size_t i = 0; i < matrix.col_size(); ++i) {
        row.emplace_back(matrix(n_row, i));
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Row<T, I, J>::operator=(const std::vector<T>& new_row)
    -> Row<T, I, J>&
{
    if (not(row.size() == new_row.size())) {
        throw std::invalid_argument{ "Row::invalid size" };
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
    if (col > (matrix.col_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid col number" };
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
    for (std::size_t i = 0; i < matrix.col_size(); ++i) {
        row.emplace_back(matrix(n_row, i));
    }
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Crow<T, I, J>::operator[](std::size_t col) const -> T
{
    if (col > (matrix.col_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid col number" };
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
    const -> T&
{
    if (row > (row_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid row number" };
    }
    if (col > (col_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid col number" };
    }

    return data[row][col];
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::at(std::size_t row, std::size_t col) const -> T&
{
    if (row > (row_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid row number" };
    }
    if (col > (col_size() - 1)) {
        throw std::out_of_range{ "Matrix::invalid col number" };
    }

    return data[row][col];
}

template <detail::Arithmetic U, std::size_t A, std::size_t B>
constexpr auto operator<<(std::ostream& ostream, const Matrix<U, A, B>& matrix)
    -> std::ostream&
{
    for (std::size_t i = 0; i < matrix.size_.first; ++i) {
        for (std::size_t j = 0; j < matrix.size_.second; ++j) {
            ostream << matrix.data[i][j] << " ";
        }
        ostream << "\n";
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
    std::size_t col_) noexcept
    : matrix{ matrix_ }, row{ row_ }, col{ col_ }
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::const_iterator::const_iterator(
    const Matrix<T, I, J>& matrix_,
    std::size_t row_,
    std::size_t col_) noexcept
    : matrix{ matrix_ }, row{ row_ }, col{ col_ }
{
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator*() noexcept -> T&
{
    return matrix.underlying_array()[row][col];
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator*() const noexcept -> const T&
{
    return matrix.underlying_array()[row][col];
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator*() const noexcept -> const T&
{
    if (row >= matrix.row_size() or col >= matrix.col_size()) {
        static T default_value{};
        return default_value;
    }
    return matrix(row, col);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++() noexcept
    -> Matrix<T, I, J>::iterator&
{
    ++col;
    if (col == matrix.col_size()) [[unlikely]] {
        col = 0;
        ++row;
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator++() noexcept
    -> Matrix<T, I, J>::const_iterator&
{
    ++col;
    if (col == matrix.col_size()) [[unlikely]] {
        col = 0;
        ++row;
    }

    return *this;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++(int) noexcept
    -> Matrix<T, I, J>::iterator
{
    auto temp = *this;

    ++(*this);

    return temp;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator++(int) noexcept
    -> Matrix<T, I, J>::const_iterator
{
    auto temp = *this;

    ++(*this);

    return temp;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
inline auto Matrix<T, I, J>::iterator::operator==(
    const iterator& iter) const noexcept -> bool
{
    return row == iter.row and col == iter.col;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
inline auto Matrix<T, I, J>::const_iterator::operator==(
    const const_iterator& iter) const noexcept -> bool
{
    return row == iter.row and col == iter.col;
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator!=(const iterator& iter) const noexcept
    -> bool
{
    return not(*this == iter);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::const_iterator::operator!=(
    const const_iterator& iter) const noexcept -> bool
{
    return not(*this == iter);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::begin() noexcept -> Matrix<T, I, J>::iterator
{
    return iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::begin() const noexcept -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, 0, 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::end() noexcept -> Matrix<T, I, J>::iterator
{
    return iterator(*this, row_size(), 0);
}

template <detail::Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::end() const noexcept -> Matrix<T, I, J>::const_iterator
{
    return const_iterator(*this, row_size(), 0);
}

}  // namespace mtl

#endif  // MTL_MATRIX_HPP