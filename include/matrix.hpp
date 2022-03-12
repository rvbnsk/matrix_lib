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
    void insert(const int &);
    Matrix operator+(const Matrix &);
    // Matrix operator*(const Matrix &);
    friend std::ostream &operator<<(std::ostream &, const Matrix &);
};

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix()
{
    std::cout << "allocation" << std::endl;
    array = new T *[I];
    for (int i = 0; i < I; ++i) { array[i] = new T[J]; }
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::~Matrix()
{
    for (int i = 0; i < I; ++i) { delete[] array[i]; }
    delete array;

    array = nullptr;
}

template <typename T, std::size_t I, std::size_t J>
void Matrix<T, I, J>::insert(const int &data)
{
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) { this->array[i][j] = data; }
    }
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J> Matrix<T, I, J>::operator+(const Matrix<T, I, J> &array)
{
    Matrix result;
    for (auto i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            result.array[i][j] = this->array[i][j] + array.array[i][j];
        }
    }

    return result;
}

template <typename T, std::size_t I, std::size_t J>
std::ostream &operator<<(std::ostream &os, const Matrix<T, I, J> &array)
{
    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) { os << array.array[i][j] << " "; }
        os << std::endl;
    }

    return os;
}

}  // namespace mtl

#endif  // MTL_MATRIX_HPP