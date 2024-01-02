#include <ostream>
#include <iostream>
#include <vector>
#include "LinearAlgebra.hpp"
#include "utils.hpp"

#include <chrono>
using namespace std::chrono;

using matrix = std::vector<std::vector<double>>;



int main(int argc, char *argv[])
{
    matrix A = matrix(10, std::vector<double>{303, 123, 414, 141, 111, 55, 303, 123, 414, 141, 111, 55, 303, 123, 414, 141, 111, 55});
    matrix B = matrix(18, std::vector<double>{1,2,3,4,5,6,7,8,9,10});
    matrix C = matMul(A, B);
    
    printMatrix(C, "C");
    return 0;
}