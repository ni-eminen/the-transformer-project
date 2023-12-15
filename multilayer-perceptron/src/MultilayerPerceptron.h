#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h> 

class MultilayerPerceptron {
public:
    double learning_rate;
    double bias;
    std::std::vector<double> weights;

    MultilayerPerceptron(double bias, double learning_rate, std::std::vector<double> initial_weights);

    void print(std::string x);

    double weightedSum(std::std::vector<double> weights, std::std::vector<double> inputs);

    double sigmoid(double x);

    double dSigmoid(double x);

    double combinationFunction(std::std::vector<double> weights, std::std::vector<double> inputs);

    double activationFunction(double x);

    double forward(std::std::vector<double> inputs);

    void train(std::std::vector<double> x, std::std::vector<double> y);

    double lossFunction(std::std::vector<double> y, std::std::vector<double> y_pred);
};

#endif // PERCEPTRON_H