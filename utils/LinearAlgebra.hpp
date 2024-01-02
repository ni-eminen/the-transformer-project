#include <ostream>
#include <vector>
#include <string>
using matrix = std::vector<std::vector<double>>;

matrix matMul(matrix A, matrix B);

void printMatrix(matrix m, std::string label);