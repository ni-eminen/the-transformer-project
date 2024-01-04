#include <iostream>
#include <vector>
#include <string>
#include <cfloat>
#include <cmath>
#include "utils.hpp"
#include "Types.hpp"

void printVector(vector<double> v, std::string title = "vector") {
    std::cout << std::endl << title << ": [";
    for(double x : v) {
        std::cout << x << ", ";
    }
    std::cout << "]" << std::endl;
}

vector<double> replaceZeros(vector<double> vec, double replacement) {
    for(int i = 0; i<vec.size(); i++) {
        if (vec[i] == 0.0) {
            vec[i] = replacement;
        }
    }

    return vec;
}

double binary_cross_entropy(vector<double> y, vector<double> y_pred) {
    y = replaceZeros(y, DBL_MIN);
    y_pred = replaceZeros(y_pred, DBL_MIN);

    double result = 0;
    for (int i = 0; i<y.size(); i++) {
        if (y_pred[i] == 1) {
            y_pred[i] = 1.0 - 1.0/pow(10, 10);
        }

        double temp = y[i] * log(y_pred[i]) + (1.0-y[i])*(log(1.0-y_pred[i]));
        result = result + temp;
    }

    return -(1.0/y.size()) * result;
}



double d_binary_cross_entropy(vector<double> y, vector<double> y_pred) {
    y = replaceZeros(y, DBL_MIN);
    y_pred = replaceZeros(y_pred, DBL_MIN);

    double result = 0;
    for(int i = 0; i<y.size(); i++) {
        if (y_pred[i] == 1) {
            y_pred[i] = 1.0 - 1.0/pow(10, 10);
        }

        double temp = (y[i] / y_pred[i]) - ((1.0 - y[i]) / (1.0 - y_pred[i]));
        result = result + temp;
    }

    return -(1.0/y.size()) * result;
}


double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double dSigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double weightedSum(vector<double> weights, vector<double> inputs) {
    double result = 0;

    for(int i = 0; i<weights.size(); i++) {
        result += (weights[i] * inputs[i]);
    }

    return result;
}