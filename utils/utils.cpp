#include <iostream>
#include <vector>
#include <string>
#include <cfloat>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "utils.hpp"
#include "Types.hpp"

void printVector(vector<double> v, std::string title = "vector")
{
    std::cout << std::endl
              << title << "[";
    for (double x : v)
    {
        std::cout << x << ", ";
    }
    std::cout << "]" << std::endl;
}

vector<double> replaceZeros(vector<double> vec, double replacement)
{
    for (int i = 0; i < vec.size(); i++)
    {
        if (vec[i] == 0.0)
        {
            vec[i] = replacement;
        }
    }

    return vec;
}

double binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred)
{
    if (y.size() != y_pred.size())
    {
        throw std::invalid_argument("Sizes of y and y_pred must match");
    }

    if (y.empty())
    {
        return 0.0;
    }

    double epsilon = 1e-10;
    double result = 0;
    for (size_t i = 0; i < y.size(); i++)
    {
        double y_pred_clamped = std::max(epsilon, std::min(1.0 - epsilon, y_pred[i]));

        result += y[i] * log(y_pred_clamped) + (1.0 - y[i]) * log(1.0 - y_pred_clamped);
    }

    return -(1.0 / y.size()) * result;
}

double d_binary_cross_entropy(vector<double> y, vector<double> y_pred)
{
    double epsilon = 1e-10; // A small constant to avoid division by zero

    double result = 0.0;
    for (size_t i = 0; i < y.size(); i++)
    {
        double y_pred_clamped = std::max(epsilon, std::min(1.0 - epsilon, y_pred[i])); // Clamping y_pred to avoid extreme values

        // Calculating the binary cross-entropy derivative
        result += (y[i] / y_pred_clamped) - ((1.0 - y[i]) / (1.0 - y_pred_clamped));
    }

    return -result / y.size(); // Averaging and negating the result
}

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double dSigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

double RELU_CEILING = 2;

double ReLU(double x)
{
    if (x >= RELU_CEILING)
    {
        return RELU_CEILING;
    }
    return std::max(0.0, x);
}

double dReLU(double x)
{
    if (x > 0.0 && x <= RELU_CEILING)
        return 1;
    if (x > RELU_CEILING)
        return 0;
    return 0;
}

double linear(double x)
{
    return x;
}

double dLinear(double x)
{
    return 1;
}

double weightedSum(const std::vector<double> &weights, const std::vector<double> &inputs)
{
    // Check for size mismatch
    if (weights.size() != inputs.size())
    {
        throw std::invalid_argument("Sizes of weights and inputs must match");
    }

    // Return 0 for empty vectors
    if (weights.empty())
    {
        return 0.0;
    }

    // Calculate the dot product
    return std::inner_product(weights.begin(), weights.end(), inputs.begin(), 0.0);
}

void printMatrix(const std::vector<std::vector<double>> &A, const std::string &title)
{
    std::cout << title << std::endl;

    if (A.empty())
    {
        std::cout << "(empty matrix)" << std::endl;
    }
    else
    {
        for (size_t i = 0; i < A.size(); i++)
        {
            printVector(A[i], "");
        }
    }

    std::cout << "-------------------------------" << std::endl;
}

vector<double> positionalEncoding(vector<double> v, int position, int d_model)
{
    vector<double> result;

    for (int i = 0; i < v.size(); i++)
    {
        bool even = i % 2 == 0;

        if (even)
        {
            result.push_back(std::sin(position / std::pow(10000, (2 * i) / d_model)));
        }
        else
        {
            result.push_back(std::cos(position / std::pow(10000, (2 * i) / d_model)));
        }
    }

    return result;
}
