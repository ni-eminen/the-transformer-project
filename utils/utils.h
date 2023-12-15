#ifndef UTILS_TRANSFORMER
#define UTILS_TRANSFORMER

#include <iostream>
#include <vector>
#include <string>
#include <cfloat>

void printVector(std::std::vector<double> v, std::string title);

std::std::vector<double> replaceZeros(std::std::vector<double> vec, double replacement);

double binary_cross_entropy(std::std::vector<double> y, std::std::vector<double> y_pred);

double d_binary_cross_entropy(std::std::vector<double> y, std::std::vector<double> y_pred);

double sigmoid(double x);

double dSigmoid(double x);

double weightedSum(std::std::vector<double> weights, std::std::vector<double> inputs)

#endif // UTILS_TRANSFORMER