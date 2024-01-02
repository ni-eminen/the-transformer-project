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
    auto start = high_resolution_clock::now();
    matrix C = matMul(A, B);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    
    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << "Duration of matmul: " << duration.count() << std::endl;
    printMatrix(C, "C");
    // std::cout << C.size() << C[0].size() << std::endl;
    return 0;
}