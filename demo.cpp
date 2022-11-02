#include <iostream>
#include "include/matrix.hpp"

int main()
{
    const mtl::Matrix<int, 3, 3> m1{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const mtl::Matrix<int, 3, 3> m2{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    std::cout << m1 << std::endl << m2 << std::endl;

    const auto result = m1.transpoze();

    std::cout << result << std::endl;

    const auto result2 = m1 + m2;

    std::cout << result2 << std::endl;

    for(const auto& elem : m2)
    {
        std::cout << elem << std::endl;
    }

    return 0;
}