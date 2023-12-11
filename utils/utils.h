#ifndef UTILS_TRANSFORMER
#define UTILS_TRANSFORMER

#include <iostream>
#include <vector>
#include <string>
#include <cfloat>

void printVector(std::vector<double> v, std::string title = "vector");

std::vector<double> replaceZeros(std::vector<double> vec, double replacement);

double binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred);

double d_binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred);

#endif // UTILS_TRANSFORMER