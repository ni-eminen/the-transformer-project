#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>

class Perceptron {
    public:
        double learningRate;
        double bias;
        std::vector<double> weights;

        Perceptron(double bias, double learning_rate, std::vector<double> initial_weights);

        double combinationFunction(std::vector<double> weights, std::vector<double> inputs);

        double activationFunction(double x);

        double forward(std::vector<double> inputs);

        void train(std::vector<double> x, std::vector<double> y);

        double lossFunction(std::vector<double> y, std::vector<double> y_pred);
};

#endif // PERCEPTRON_H