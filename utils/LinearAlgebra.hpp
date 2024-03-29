#include <ostream>
#include <vector>
#include <string>
#include "Types.hpp"

vector<vector<double> > matMul(vector<vector<double> > A, vector<vector<double> > B);
vector<vector<double> > matMul(std::vector<double> A, vector<vector<double> > B);
vector<vector<double> > matMul(vector<vector<double> > A, std::vector<double> B);
double matMul(std::vector<double> A, std::vector<double> B);
double sumVector(vector<double> v);
double expSumVector(vector<double> v);
vector<vector<double> > transpose(vector<vector<double> > A);
vector<double> elementWiseSum(vector<double> a, vector<double> b);

void printVector(vector<vector<double> > m, std::string label);

vector<vector<double> > axbVector(int a, int b);

vector<vector<double> > scalarMultiplyMatrix(double scalar, vector<vector<double> > A);

vector<double> vectorAddition(vector<double> v1, vector<double> v2);

vector<double> vectorNormalization(vector<double> v);
