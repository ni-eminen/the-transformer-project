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
    matrix A = matrix(3, std::vector<double>{1,2,3});
    matrix B = matrix(3, std::vector<double>{1,2,3});
    matrix C = matMul(A, B);
    
    printMatrix(C, "C");
    
    printMatrix(transpose(C), "C transposed");


    return 0;
}