#ifndef UTILS_TRANSFORMER
#define UTILS_TRANSFORMER

#include <iostream>
#include <vector>
#include <string>
#include <cfloat>
#include "Types.hpp"

void printVector(vector<double> v, std::string title);

void printMatrix(const std::vector<std::vector<double> > &A, const std::string &title = "matrix");

std::vector<double> replaceZeros(std::vector<double> vec, double replacement);

double binary_cross_entropy(vector<double> y, vector<double> y_pred);

double d_binary_cross_entropy(vector<double> y, vector<double> y_pred);

double sigmoid(double x);

double dSigmoid(double x);

double ReLU(double x);

int dReLU(double x);

double weightedSum(const std::vector<double> &weights, const std::vector<double> &inputs);

#endif // UTILS_TRANSFORMER