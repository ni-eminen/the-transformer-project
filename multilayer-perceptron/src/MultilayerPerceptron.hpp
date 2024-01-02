#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>

std::vector<std::vector<double>> generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue);

class MultilayerPerceptron {
public:
    double learningRate;
    std::vector<std::vector<double>> inputWeights;
    std::vector<std::vector<double>> hiddenWeights;
    std::vector<std::vector<double>> outputWeights;
    std::vector<double> hiddenBiases;
    std::vector<double> outputBiases;
    int hiddenLayerDim;
    int inputLayerDim;
    int outputLayerDim;

    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;

    MultilayerPerceptron(double initialBias, 
                         double initialWeightValue, 
                         int hiddenLayerDim, 
                         int inputLayerDim, 
                         int outputLayerDim, 
                         double learningRate);

    void print(std::string x);

    double combinationFunction(std::vector<double> weights, std::vector<double> inputs);

    double activationFunction(double x);

    std::vector<double> forward(std::vector<double> inputs);

    void train(std::vector<double> x, std::vector<double> y);

    double lossFunction(std::vector<double> y, std::vector<double> y_pred);
};

#endif // PERCEPTRON_H