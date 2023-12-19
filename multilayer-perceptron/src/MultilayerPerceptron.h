#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>

class MultilayerPerceptron {
public:
    double learningRate;
    double bias;
    std::vector<std::vector<std::vector<double>>> weights;
    int layerDim;
    int layerAmt;

    MultilayerPerceptron(double initialBias, double initialWeightValue, int layerAmount, int layerDimension, double learningRate);

    void print(std::string x);

    double sigmoid(double x);

    double combinationFunction(std::vector<double> weights, std::vector<double> inputs);

    double activationFunction(double x);

    double forward(std::vector<std::vector<std::vector<double>>> inputs);

    void train(std::vector<double> x, std::vector<double> y);

    double lossFunction(std::vector<double> y, std::vector<double> y_pred);
};

#endif // PERCEPTRON_H