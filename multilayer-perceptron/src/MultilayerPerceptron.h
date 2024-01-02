#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>

class MultilayerPerceptron {
public:
    double learningRate;
    std::vector<std::vector<double>> inputWeights;
    std::vector<std::vector<std::vector<double>>> hiddenWeights;
    std::vector<std::vector<double>> outputWeights;
    std::vector<std::vector<double>> hiddenBiases;
    std::vector<std::vector<double>> outputBiases;
    int hiddenLayerDim;
    int inputLayerDim;
    int outputLayerDim;

    MultilayerPerceptron(double initialBias, 
                         double initialWeightValue, 
                         int hiddenLayerDim, 
                         int inputLayerDim, 
                         double outputLayerDim, 
                         double learningRate);

    void print(std::string x);

    double sigmoid(double x);

    double combinationFunction(std::vector<double> weights, std::vector<double> inputs);

    double activationFunction(double x);

    double forward(std::vector<double> inputs);

    void train(std::vector<double> x, std::vector<double> y);

    double lossFunction(std::vector<double> y, std::vector<double> y_pred);
};

#endif // PERCEPTRON_H