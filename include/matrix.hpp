#ifndef MTL_MATRIX_HPP
#define MTL_MATRIX_HPP

#include <concepts>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mtl {

template <class T>
inline constexpr auto is_arithmetic_v = std::is_arithmetic<T>::value;

template <class T, class U>
inline constexpr auto is_same_v = std::is_same<T, U>::value;

template <typename T>
concept Arithmetic = is_arithmetic_v<T> &&requires(T type)
{
    type + type;
    type - type;
    type *type;
    type == type;
    type != type;
};

template <typename T, typename U>
concept Scalar = std::is_scalar_v<U> &&requires(T t, U u)
{
    t *u;
};

template <Arithmetic T, std::size_t I, std::size_t J>
class Matrix {
   private:
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
    T array[I][J];
    unsigned int fill_counter = 0;
    static constexpr std::pair<std::size_t, std::size_t> size_{I, J};

    class Row {
       private:
        Matrix<T, I, J> matrix;
        std::size_t row;

       public:
        Row(Matrix<T, I, J> &matrix, std::size_t row) : matrix(matrix), row(row)
        {
        }
        auto operator[](std::size_t col) -> T &;
        auto operator[](std::size_t col) const -> const T &;
        friend class Matrix;
    };

   public:
    Matrix();
    ~Matrix() = default;

    Matrix(const Matrix<T, I, J> &array);
    Matrix(Matrix<T, I, J> &&array) noexcept;

    template <Arithmetic U, std::size_t A, std::size_t B>
    explicit Matrix(const Matrix<U, A, B> &array);

    template <Arithmetic U, std::size_t A, std::size_t B>
    explicit Matrix(Matrix<U, A, B> &&array) noexcept;

    auto operator=(const Matrix<T, I, J> &array) -> Matrix<T, I, J> &;
    auto operator=(Matrix<T, I, J> &&array) noexcept -> Matrix<T, I, J> &;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator=(const Matrix<U, A, B> &array) -> Matrix<T, I, J> &;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator=(Matrix<U, A, B> &&array) noexcept -> Matrix<T, I, J> &;

    void insert(const T &element);
    void sort();
    auto transpoze() -> Matrix<T, I, J>;
    auto power(const unsigned int &power) -> Matrix<T, I, J> &;
    auto det() -> T;
    auto det() const -> T;
    auto is_diagonal() -> bool;
    auto is_diagonal() const -> bool;
    void ones();
    constexpr auto size() -> std::pair<std::size_t, std::size_t>;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator+(const Matrix<U, A, B> &array) -> Matrix<T, I, J>;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator+=(const Matrix<U, A, B> &array) -> Matrix<T, I, J> &;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator-(const Matrix<U, A, B> &array) -> Matrix<T, I, J>;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator-=(const Matrix<U, A, B> &array) -> Matrix<T, I, J> &;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator*(const Matrix<U, A, B> &array) -> Matrix<T, I, J>;

    template <Arithmetic U, std::size_t A, std::size_t B>
    auto operator*=(const Matrix<U, A, B> &array) -> Matrix<T, I, J> &;

    template <Scalar<T> U>
    auto operator*(const U &scalar) -> Matrix<T, I, J>;

    template <Scalar<T> U>
    auto operator*=(const U &scalar) -> Matrix<T, I, J> &;

    auto operator^(const unsigned int &power) -> Matrix<T, I, J> &;

    template <Arithmetic U, std::size_t A, std::size_t B>
    inline auto operator==(const Matrix<U, A, B> &array) const -> bool;

    template <Arithmetic U, std::size_t A, std::size_t B>
    inline auto operator!=(const Matrix<U, A, B> &array) const -> bool;

    auto operator[](std::size_t row) -> Row &;
    auto operator[](std::size_t row) const -> const Row &;
    auto operator()(std::size_t row, std::size_t col) -> T &;
    auto operator()(std::size_t row, std::size_t col) const -> const T &;

    template <Arithmetic U, std::size_t A, std::size_t B>
    friend auto operator<<(std::ostream &, const Matrix<U, A, B> &)
        -> std::ostream &;

    class iterator {
       private:
        std::size_t row;
        std::size_t col;

       public:
        iterator(std::size_t row, std::size_t col);
        auto operator*() const -> T &;
        auto operator++() -> iterator &;
        auto operator++(int) -> iterator;
        auto operator==(const iterator &iter) const -> bool;
        auto operator!=(const iterator &iter) const -> bool;
        friend class Matrix;
    }

    private : auto
              begin() const -> iterator;
    auto end() const -> iterator;
};

template <Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix()
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = 0; }
    }
}

template <Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix(const Matrix<T, I, J> &array)
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = array.array[i][j]; }
    }
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
Matrix<T, I, J>::Matrix(const Matrix<U, A, B> &array)
{
    static_assert(is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A || J == B, "Matrix::invalid size");
    *this = array;
}

template <Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix(Matrix<T, I, J> &&array) noexcept
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = array.array[i][j]; }
    }

    array.~Matrix();
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
Matrix<T, I, J>::Matrix(Matrix<U, A, B> &&array) noexcept
{
    static_assert(is_same_v<T, U>, "Matrix::invalid type");
    static_assert(I == A || J == B, "Matrix::invalid size");

    *this = array;

    array.~Matrix();
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator=(const Matrix<T, I, J> &array)
    -> Matrix<T, I, J> &
{
    if (&array != this) {
        for (auto i = 0; i < I; ++i) {
            for (auto j = 0; j < J; ++j) {
                this->array[i][j] = array.array[i][j];
            }
        }
    }

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator=(const Matrix<U, A, B> &array)
    -> Matrix<T, I, J> &
{
    static_assert(is_same_v<T, U>, "Matrix invalid type");
    static_assert(I == A || J == B, "Matrix::invalid size");

    *this = array;

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator=(Matrix<T, I, J> &&array) noexcept
    -> Matrix<T, I, J> &
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = array.array[i][j]; }
    }

    array.~Matrix();

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator=(Matrix<U, A, B> &&array) noexcept
    -> Matrix<T, I, J> &
{
    static_assert(is_same_v<T, U>, "Matrix invalid type");
    static_assert(I == A || J == B, "Matrix::invalid size");

    *this = array;

    array.~Matrix();

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
void Matrix<T, I, J>::insert(const T &element)
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = element; }
    }
}

template <Arithmetic T, std::size_t I, std::size_t J>
void Matrix<T, I, J>::sort()
{
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::transpoze() -> Matrix<T, I, J>
{
    Matrix<T, I, J> result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { result.array[i][j] = this->array[i][j]; }
    }

    return result;
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::det() -> T
{
    static_assert(I == J, "Matrix::det::invalid size");

    T det = 0;
    int sign = -1;

    Matrix temp;

    if (I == 1) { return this->array[0][0]; }

    if (I == 2) {
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

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::det() const -> T
{
    static_assert(I == J, "Matrix::det::invalid size");

    T det = 0;
    int sign = -1;

    Matrix temp;

    if (I == 1) { return this->array[0][0]; }

    if (I == 2) {
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

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::is_diagonal() -> bool
{
    if (I != J) { return false; }

    for (auto m = 0; m < I; ++m) {
        for (auto n = 0; n < I; ++n) {
            if (m != n && this->array[m][n] != 0) { return false; }
        }
    }

    return true;
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::is_diagonal() const -> bool
{
    if (I != J) { return false; }

    for (auto m = 0; m < I; ++m) {
        for (auto n = 0; n < I; ++n) {
            if (m != n && this->array[m][n] != 0) { return false; }
        }
    }

    return true;
}

template <Arithmetic T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::size() -> std::pair<std::size_t, std::size_t>
{
    return size_;
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::power(const unsigned int &power) -> Matrix<T, I, J> &
{
    for (auto i = 0; i < (power - 1); ++i) { *this *= *this; }

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator+(const Matrix<U, A, B> &array) -> Matrix<T, I, J>
{
    static_assert(I == A || J == B, "Matrix::invalid size");
    static_assert(is_same_v<T, U>, "Matrix::invalid type");

    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            result.array[i][j] = this->array[i][j] + array.array[i][j];
        }
    }

    return result;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator+=(const Matrix<U, A, B> &array)
    -> Matrix<T, I, J> &
{
    static_assert(I == A || J == B, "Matrix::invalid size");
    static_assert(is_same_v<T, U>, "Matrix::invalid type");

    *this = *this + array;

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator-(const Matrix<U, A, B> &array) -> Matrix<T, I, J>
{
    static_assert(I == A || J == B, "Matrix::invalid size");
    static_assert(is_same_v<T, U>, "Matrix::invalid type");

    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            result.array[i][j] = this->array[i][j] - array.array[i][j];
        }
    }

    return result;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator-=(const Matrix<U, A, B> &array)
    -> Matrix<T, I, J> &
{
    static_assert(I == A || J == B, "Matrix::invalid size");
    static_assert(is_same_v<T, U>, "Matrix::invalid type");

    *this = *this - array;

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator*(const Matrix<U, A, B> &array) -> Matrix<T, I, J>
{
    static_assert(I == B, "Matrix::invalid size");
    static_assert(is_same_v<T, U>, "Matrix::invalid type");

    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            result.array[i][j] = 0;
            for (auto k = 0; k < J; ++k) {
                result.array[i][j] += this->array[i][k] * array.array[k][j];
            }
        }
    }

    return result;
}

template <Arithmetic T, std::size_t I, size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator*=(const Matrix<U, A, B> &array)
    -> Matrix<T, I, J> &
{
    static_assert(I == B, "Matrix::invalid size");
    static_assert(is_same_v<T, U>, "Matrix::invalid type");

    *this = *this * array;

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Scalar<T> U>
auto Matrix<T, I, J>::operator*(const U &scalar) -> Matrix<T, I, J>
{
    Matrix result;
    for (auto i = 0; i < J; ++i) {
        for (auto j = 0; j < J; ++j) {
            result.array[i][j] = this->array[i][j] * scalar;
        }
    }

    return result;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Scalar<T> U>
auto Matrix<T, I, J>::operator*=(const U &scalar) -> Matrix<T, I, J> &
{
    *this = *this * scalar;

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator^(const unsigned int &power) -> Matrix<T, I, J> &
{
    this->power(power);

    return *this;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
inline auto Matrix<T, I, J>::operator==(const Matrix<U, A, B> &array) const
    -> bool
{
    if (I != A || J != B || typeid(T).name() != typeid(U).name()) {
        return false;
    }

    for (auto i = 0; i < J; ++i) {
        for (auto j = 0; j < J; ++j) {
            if (this->array[i][j] != array.array[i][j]) { return false; }
        }
    }

    return true;
}

template <Arithmetic T, std::size_t I, std::size_t J>
template <Arithmetic U, std::size_t A, std::size_t B>
inline auto Matrix<T, I, J>::operator!=(const Matrix<U, A, B> &array) const
    -> bool
{
    return !(*this == array);
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator[](std::size_t row) -> Matrix<T, I, J>::Row &
{
    return Row(*this, row);
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator[](std::size_t row) const
    -> const Matrix<T, I, J>::Row &
{
    return Row(*this, row);
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::Row::operator[](std::size_t col) -> T &
{
    return matrix(row, col);
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::Row::operator[](std::size_t col) const -> const T &
{
    return matrix(row, col);
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col) -> T &
{
    return this->array[row][col];
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col) const
    -> const T &
{
    return this->array[row][col];
}

template <Arithmetic U, std::size_t A, std::size_t B>
auto operator<<(std::ostream &os, const Matrix<U, A, B> &array)
    -> std::ostream &
{
    for (auto i = 0; i < A; ++i) {
        for (auto j = 0; j < B; ++j) { os << array.array[i][j] << " "; }
        os << std::endl;
    }

    return os;
}

template <Arithmetic T, std::size_t I, std::size_t J>
Matrix<T, I, J>::iterator::iterator(std::size_t row_, std::size_t col_)
    : row(row_), col(col_)
{
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator*() const -> T &
{
    return array[row][col];
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++() -> Matrix<T, I, J>::iterator &
{
}

template <Arithmetic T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::iterator::operator++(int) -> Matrix<T, I, J>::iterator
{
}

}  // namespace mtl

#endif  // MTL_MATRIX_HPP