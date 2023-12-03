#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <math.h> 

class Neuron {
public:
    double learning_rate;
    double bias;
    std::vector<double> weights;

    Neuron(double bias, double learning_rate, std::vector<double> initial_weights);

    void print(std::string x);

    double weighted_sum(std::vector<double> weights, std::vector<double> inputs);

    double sigmoid(double x);

    double d_sigmoid(double x);

    double combination_function(std::vector<double> weights, std::vector<double> inputs);

    double activation_function(double x);

    double forward_propagate(std::vector<double> inputs);

    void train(std::vector<double> x, std::vector<double> y);

    double loss_function(std::vector<double> y, std::vector<double> y_pred);
};

#endif // NEURON_H