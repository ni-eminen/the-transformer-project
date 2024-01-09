#include <ostream>
#include <iostream>
#include <vector>
#include "LinearAlgebra.hpp"
#include "utils.hpp"

#include <chrono>
using namespace std::chrono;

using vector<vector<double> > = std::vector<std::vector<double> >;

int main(int argc, char *argv[])
{
    vector<vector<double> > A = vector<vector<double> >(3, std::vector<double>{1, 2, 3});
    vector<vector<double> > B = vector<vector<double> >(3, std::vector<double>{1, 2, 3});
    vector<vector<double> > C = matMul(A, B);

    printVector<vector<double> >(transpose(C), "C transposed");

    return 0;
}