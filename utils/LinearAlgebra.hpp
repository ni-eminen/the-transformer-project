#include <ostream>
#include <vector>
#include <string>
#include "Types.hpp"

matrix matMul(matrix A, matrix B);
vector<double> matMul(std::vector<double> A, matrix B);
vector<double> matMul(matrix A, std::vector<double> B);
vector<double> matMul(std::vector<double> A, std::vector<double> B);

matrix transpose(matrix A);
vector<double> elementWiseSum(vector<double> a, vector<double> b);

void printMatrix(matrix m, std::string label);