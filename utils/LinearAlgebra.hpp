#include <ostream>
#include <vector>
#include <string>
using matrix = std::vector<std::vector<double>>;
using std::vector;

matrix matMul(matrix A, matrix B);
vector<double> matMul(std::vector<double> A, matrix B);
vector<double> matMul(matrix A, std::vector<double> B);
vector<double> matMul(std::vector<double> A, std::vector<double> B);

matrix transpose(matrix A);

void printMatrix(matrix m, std::string label);