#ifndef MTL_MATRIX_HPP
#define MTL_MATRIX_HPP

#include <iostream>
#include <type_traits>
#include <utility>

namespace mtl {

template <typename T, std::size_t I, std::size_t J>
class Matrix {
   private:
    T **array = nullptr;
    unsigned int fill_counter = 0;

   public:
    Matrix();
    ~Matrix();
    Matrix(const Matrix<T, I, J> &) noexcept;
    Matrix(Matrix<T, I, J> &&) noexcept;
    constexpr auto operator=(const Matrix<T, I, J> &) noexcept
        -> Matrix<T, I, J> &;
    constexpr auto operator=(Matrix<T, I, J> &&) noexcept -> Matrix<T, I, J> &;
    void insert(const T &);
    auto transpoze() -> Matrix<T, I, J> &;

    template <typename U, std::size_t A, std::size_t B>
    Matrix<T, I, J> operator+(const Matrix<U, A, B> &);

    template <typename U, std::size_t A, std::size_t B>
    Matrix<T, I, J> &operator+=(const Matrix<U, A, B> &);

    template <typename U, std::size_t A, std::size_t B>
    Matrix<T, I, J> operator*(const Matrix<U, A, B> &);

    template <typename U, std::size_t A, std::size_t B>
    Matrix<T, I, J> &operator*=(const Matrix<U, A, B> &);

    Matrix<T, I, J> operator*(const T &);
    Matrix<T, I, J> &operator*=(const T &);

    template <typename U, std::size_t A, std::size_t B>
    inline auto operator==(const Matrix<U, A, B> &) -> bool;

    template <typename U, std::size_t A, std::size_t B>
    inline auto operator!=(const Matrix<U, A, B> &) -> bool;

    template <typename U, std::size_t A, std::size_t B>
    friend std::ostream &operator<<(std::ostream &, const Matrix<U, A, B> &);
};

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix()
{
    array = new T *[I];
    for (auto i = 0; i < I; ++i) { array[i] = new T[J]; }
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::~Matrix()
{
    for (auto i = 0; i < I; ++i) { delete[] array[i]; }
    delete array;

    array = nullptr;
}

template <typename T, std::size_t I, std::size_t J>
constexpr auto Matrix<T, I, J>::operator=(Matrix<T, I, J> &&array) noexcept
    -> Matrix<T, I, J> &
{
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { this->array[i][j] = array.array[i][j]; }
    }
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
auto Matrix<T, I, J>::transpoze() -> Matrix<T, I, J> &
{
    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) { result.array[i][j] = this->array[i][j]; }
    }
    return result;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> Matrix<T, I, J>::operator+(const Matrix<U, A, B> &array)
{
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
    for (auto i = 0; i < I; ++i) {
        for (auto j = 0; j < J; ++j) {
            this->array[i][j] = this->array[i][j] + array.array[i][j];
        }
    }

    return *this;
}

template <typename T, std::size_t I, std::size_t J>
template <typename U, std::size_t A, std::size_t B>
Matrix<T, I, J> Matrix<T, I, J>::operator*(const Matrix<U, A, B> &array)
{
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
template <typename U, std::size_t A, std::size_t B>
inline auto Matrix<T, I, J>::operator==(const Matrix<U, A, B> &array) -> bool
{
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