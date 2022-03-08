#include "../include/matrix.hpp"

namespace mtl {

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::Matrix()
{
    array = new T *[I];
    for (auto i{0}; i < I; ++i) { array[k] = new T[J]; }
}

template <typename T, std::size_t I, std::size_t J>
Matrix<T, I, J>::~Matrix()
{
    for (auto i{0}; i < I; ++i) { delete[] array[i]; }
    delete array;

    array = nullptr;
}

template <typename T, std::size_t I, std::size_t J>
std::ostream &operator<<(std::ostream &os, const Matrix<T, I, J> &array)
{
    return os;
}

}  // namespace mtl