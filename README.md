# matrix_lib
C++ library for matrix storage and easy management.

Template Matrix Container provides easy to use matrix objet.
Matrices can be managed using provided mathematical operations and functions.

**Examples of use:**

**Addition:**
```C++
mtl::Matrix<int, 2, 2> matrix { {1, 2},
                                {3, 4} };
mtl::Matrix<int, 2, 2> matrix2(5);

const auto result = matrix + matrix2;
```
Result:

Matrix of type:
```C++
mtl::Matrix<int, 2, 2>
```
With given values:
```
15 15
35 35
```

**Determinant:**
```C++
mtl::Matrix<unsigned int, 3, 3> matrix { {7, 2, 9},
                                         {4, 5, 3},
                                         {2, 6, 7} };
matrix.det();
```
Result:
```
201
```

**Transposition:**
```C++
mtl::Matrix<double, 2, 3> matrix { {1, 2, 3},
                                   {4, 5, 6} };
const auto result = matrix.transpose();
```
Result:

Matrix of type:
```C++
mtl::Matrix<double, 3, 2>
```
With given values:
```
1 4
2 5
3 6
```
