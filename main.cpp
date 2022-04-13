#include <iostream>
#include "include/matrix.hpp"

int main()
{
    mtl::Matrix<int, 3, 3> m1;
    m1.insert(2);
    auto m2 = m1;
    
    mtl::Matrix<int, 2, 2> m3{1, 1, 1, 1};
    std::cout << (m1 == m2);
    //m1.power(2);
    //std::cout << m1;
    return 0;
}