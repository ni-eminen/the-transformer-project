#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h> 

class NeuralNetwork {
public:
    double learningRate;
    std::std::vector<double> weights;

    NeuralNetwork(double learningRate, std::std::vector<std::std::vector<std::std::vector<double>>> initialWeights);

    double combinationFunction(std::std::vector<double> weights, std::std::vector<double> inputs);

    double activationFunction(double x);
    
    double dActivationFunction(double x);

    double forward(std::std::vector<double> inputs);

    void train(std::std::vector<double> x, std::std::vector<double> y);

    double lossFunction(std::std::vector<double> y, std::std::vector<double> y_pred);
};

#endif // NEURAL_NETWORK_H