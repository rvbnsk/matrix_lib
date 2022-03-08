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
    insert(const int &);
    friend std::ostream &operator<<(std::ostream &, const Matrix &);
};

}  // namespace mtl

#endif