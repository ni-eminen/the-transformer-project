#ifndef UTILS_TRANSFORMER
#define UTILS_TRANSFORMER

#include <iostream>
#include <vector>
#include <string>
#include <cfloat>
#include "Types.hpp"

void printVector(vector<double> v, std::string title);
void printMatrix(vector<vector<double> > A, std::string title);

std::vector<double> replaceZeros(std::vector<double> vec, double replacement);

double binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred);

double d_binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred);

double sigmoid(double x);

double dSigmoid(double x);

double weightedSum(std::vector<double> weights, std::vector<double> inputs);

#endif // UTILS_TRANSFORMER