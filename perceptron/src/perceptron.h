#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h> 

class Perceptron {
public:
    double learning_rate;
    double bias;
    std::vector<double> weights;

    Perceptron(double bias, double learning_rate, std::vector<double> initial_weights);

    void print(std::string x);

    double weightedSum(std::vector<double> weights, std::vector<double> inputs);

    double sigmoid(double x);

    double dSigmoid(double x);

    double combinationFunction(std::vector<double> weights, std::vector<double> inputs);

    double activationFunction(double x);

    double forwardPropagate(std::vector<double> inputs);

    void train(std::vector<double> x, std::vector<double> y);

    double lossFunction(std::vector<double> y, std::vector<double> y_pred);
};

#endif // PERCEPTRON_H