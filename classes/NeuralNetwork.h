#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h> 

class NeuralNetwork {
public:
    double learningRate;
    std::vector<double> weights;

    NeuralNetwork(double learningRate, std::vector<std::vector<std::vector<double>>> initialWeights);

    double combinationFunction(std::vector<double> weights, std::vector<double> inputs);

    double activationFunction(double x);
    
    double dActivationFunction(double x);

    double forward(std::vector<double> inputs);

    void train(std::vector<double> x, std::vector<double> y);

    double lossFunction(std::vector<double> y, std::vector<double> y_pred);
};

#endif // NEURAL_NETWORK_H