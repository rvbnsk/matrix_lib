#ifndef MTL_MATRIX_HPP
#define MTL_MATRIX_HPP

#include <concepts>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mtl {

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T> &&requires(T type)
{
    type + type;
    type *type;
};

template <typename T, std::size_t I, std::size_t J>
class Matrix {
   private:
    T array[I][J];
    unsigned int fill_counter = 0;

   private:
    class Row {
       private:
        std::size_t row;

       public:
        Row(std::size_t row) : row(row) {}
        auto operator[](std::size_t col) -> T &;
        auto operator[](std::size_t col) const -> const T &;
        friend class Matrix;
    };

   public:
    Matrix();
    ~Matrix();

    template <typename U, std::size_t A, std::size_t B>
    Matrix(const Matrix<U, A, B> &);

    template <typename U, std::size_t A, std::size_t B>
    Matrix(Matrix<T, I, J> &&);

    template <typename U, std::size_t A, std::size_t B>
    auto operator=(const Matrix<U, A, B> &) -> Matrix<T, I, J> &;

    template <typename U, std::size_t A, std::size_t B>
    auto operator=(Matrix<U, A, B> &&) -> Matrix<T, I, J> &;
    void insert(const T &);
    void sort();
    auto transpoze() -> Matrix<T, I, J> &;
    auto power(const unsigned int &) -> Matrix<T, I, J> &;
    auto det() -> T;
    auto is_diagonal() -> bool;
    auto ones();

    template <typename U, std::size_t A, std::size_t B>
    auto operator+(const Matrix<U, A, B> &) -> Matrix<T, I, J>;

    template <typename U, std::size_t A, std::size_t B>
    auto operator+=(const Matrix<U, A, B> &) -> Matrix<T, I, J> &;

    template <typename U, std::size_t A, std::size_t B>
    auto operator-(const Matrix<U, A, B> &) -> Matrix<T, I, J>;

    template <typename U, std::size_t A, std::size_t B>
    auto operator-=(const Matrix<U, A, B> &) -> Matrix<T, I, J> &;

    template <typename U, std::size_t A, std::size_t B>
    auto operator*(const Matrix<U, A, B> &) -> Matrix<T, I, J>;

    template <typename U, std::size_t A, std::size_t B>
    auto operator*=(const Matrix<U, A, B> &) -> Matrix<T, I, J> &;

    auto operator*(const T &) -> Matrix<T, I, J>;
    auto operator*=(const T &) -> Matrix<T, I, J> &;
    auto operator^(const unsigned int &) -> Matrix<T, I, J> &;

    template <typename U, std::size_t A, std::size_t B>
    inline auto operator==(const Matrix<U, A, B> &) -> bool;

    template <typename U, std::size_t A, std::size_t B>
    inline auto operator!=(const Matrix<U, A, B> &) -> bool;

    auto operator[](std::size_t) -> Row &;
    auto operator[](std::size_t) const -> const Row &;
    auto operator()(std::size_t row, std::size_t col) -> T &;

    template <typename U, std::size_t A, std::size_t B>
    friend std::ostream &operator<<(std::ostream &, const Matrix<U, A, B> &);
};

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix()
{
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::~Matrix()
{
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J>::Matrix(const Matrix<U, A, B> &array)
{
    static_assert(I != A || J != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix::invalid type");

    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = array.array[i][j]; }
    }
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator=(const Matrix<U, A, B> &array)
    -> Matrix<T, I, J> &
{
    static_assert(I != A || J != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix invalid type");

    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = array.array[i][j]; }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
auto Matrix<T, I, J>::operator=(Matrix<U, A, B> &&array) -> Matrix<T, I, J> &
{
    static_assert(I != A || J != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix invalid type");

    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = array.array[i][j]; }
    }

    array.~Matrix();

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
void Matrix<T, I, J>::insert(const T &data)
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = data; }
    }
}

template <typename T, std::size_t I, std::size_t J>
void Matrix<T, I, J>::sort()
{
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::transpoze() -> Matrix<T, I, J> &
{
    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { result.array[i][j] = this->array[i][j]; }
    }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::det() -> T
{
    static_assert(I != J, "Matrix::det::invalid size");

    T det = 0;
    short sign = -1;

    Matrix temp;

    if (I == 1) { return this->array[0][0]; }
    else if (I == 2) {
        return (
            this->array[0][0] * this->array[1][1]
            - this->array[0][1] * this->array[1][0]);
    }
    else {
        for (auto m = 0; m < I; ++m) {
            unsigned int temp_i = 0, temp_j = 0;
            for (auto n = 1; n < I; ++n) {
                for (auto o = 0; o < I; ++o) {
                    if (o == m) continue;
                    temp->array[temp_i][temp_j] = this->array[n][o];
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
    }

    return det;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::is_diagonal() -> bool
{
    if (I != J) { return false; }
    else {
        for (auto m = 0; m < I; ++m) {
            for (auto n = 0; n < I; ++n) {
                if (m != n && this->array[m][n] != 0) { return false; }
            }
        }
    }
    return true;
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::power(const unsigned int &power) -> Matrix<T, I, J> &
{
    for (auto i = 0; i < (power - 1); ++i) { this->array *= this->array; }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> Matrix<T, I, J>::operator+(const Matrix<U, A, B> &array)
{
    static_assert(I != A || J != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix::invalid type");

    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            result.array[i][j] = this->array[i][j] + array.array[i][j];
        }
    }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> &Matrix<T, I, J>::operator+=(const Matrix<U, A, B> &array)
{
    static_assert(I != A || J != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix::invalid type");

    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            this->array[i][j] = this->array[i][j] + array.array[i][j];
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> Matrix<T, I, J>::operator-(const Matrix<U, A, B> &array)
{
    static_assert(I != A || J != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix::invalid type");

    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            result.array[i][j] = this->array[i][j] - array.array[i][j];
        }
    }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> &Matrix<T, I, J>::operator-=(const Matrix<U, A, B> &array)
{
    static_assert(I != A || J != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix::invalid type");

    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            this->array[i][j] = this->array[i][j] - array.array[i][j];
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> Matrix<T, I, J>::operator*(const Matrix<U, A, B> &array)
{
    static_assert(I != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix::invalid type");

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

template <typename T, std::size_t I, size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> &Matrix<T, I, J>::operator*=(const Matrix<U, A, B> &array)
{
    static_assert(I != B, "Matrix::invalid size");
    static_assert(!std::is_same_v<T, U>, "Matrix::invalid type");

    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            this->array[i][j] = 0;
            for (auto k = 0; k < J; ++k) {
                this->array[i][j] += this->array[i][k] * array.array[k][j];
            }
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J> Matrix<T, I, J>::operator*(const T &scalar)
{
    Matrix result;
    for (auto i = 0; i < J; ++i) {
        for (auto j = 0; j < J; ++j) {
            result.array[i][j] = this->array[i][j] * scalar;
        }
    }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J> &Matrix<T, I, J>::operator*=(const T &scalar)
{
    for (auto i = 0; i < J; ++i) {
        for (auto j = 0; j < J; ++j) {
            this->array[i][j] = this->array[i][j] * scalar;
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J> &Matrix<T, I, J>::operator^(const unsigned int &power)
{
    for (auto i = 0; i < (power - 1); ++i) { this->array *= this->array; }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
inline auto Matrix<T, I, J>::operator==(const Matrix<U, A, B> &array) -> bool
{
    if (I != A || J != B) { return false; }
    else if (typeid(T).name() != typeid(U).name()) {
        return false;
    }

    for (auto i = 0; i < J; ++i) {
        for (auto j = 0; j < J; ++j) {
            if (this->array[i][j] != array.array[i][j]) return false;
        }
    }

    return true;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
inline auto Matrix<T, I, J>::operator!=(const Matrix<U, A, B> &array) -> bool
{
    return !(*this == array);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator[](std::size_t row) -> Matrix<T, I, J>::Row &
{
    return Row(row);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator[](std::size_t row) const
    -> const Matrix<T, I, J>::Row &
{
    return Row(row);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::Row::operator[](std::size_t col) -> T &
{
    return array(row, col);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::Row::operator[](std::size_t col) const -> const T &
{
    return array(row, col);
}

template <typename T, std::size_t I, std::size_t J>
auto Matrix<T, I, J>::operator()(std::size_t row, std::size_t col) -> T &
{
    return this->array[row][col];
}

template <typename U, std::size_t A, std::size_t B>
std::ostream &operator<<(std::ostream &os, const Matrix<U, A, B> &array)
{
    for (auto i = 0; i < A; ++i) {
        for (auto j = 0; j < B; ++j) { os << array.array[i][j] << " "; }
        os << std::endl;
    }

    return os;
}

}  // namespace mtl

#endif  // MTL_MATRIX_HPP