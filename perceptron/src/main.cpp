#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <math.h> 
#include <string>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <stdexcept>

class Neuron {
  public:
    double learning_rate;
    double bias;
    std::vector<double> weights;
    double activation_function(double x);
    double combination_function(std::vector<double> weights, std::vector<double> inputs);
    double predict(std::vector<double> inputs);
    double loss_function(std::vector<double> y, std::vector<double> y_pred);
    double forward_propagate(std::vector<double> x);
    std::vector<double> train(std::vector<double> x);
    Neuron(double bias, double learning_rate, std::vector<double> initial_weights) {
        this->weights = initial_weights;
        this->bias = bias;
        this->learning_rate = learning_rate;
    }


    std::vector<double> Neuron::train(std::vector<double> x) {
        double E_total = Neuron::forward_propagate(x);

        // now we must figure out for each weight, how much the weight contributed to the E_total
        for 
    }


};

double Neuron::predict(std::vector<double> inputs) {
    double weighted_sum = this->combination_function(inputs, this->weights);
    double activation_output = this->activation_function(weighted_sum);
    double pred = std::round(activation_output);

    return pred;
}

double weighted_sum(std::vector<double> weights, std::vector<double> inputs) {
    double result = 0;

    for(int i = 0; i<weights.size(); i++) {
        result += (weights[i] * inputs[i]);
    }

    return result;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double d_sigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

double Neuron::combination_function(std::vector<double> weights, std::vector<double> inputs) {
    return weighted_sum(weights, inputs);
}

double Neuron::activation_function(double x) {
    return sigmoid(x);
}

std::vector<double> replaceZeros(std::vector<double> vec, double replacement) {
    for(int i = 0; i<vec.size(); i++) {
        if (vec[i] == 0.0) {
            vec[i] = replacement;
        }
    }

    return vec;
}

double binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred) {
    // Remove zeros from arrays to avoid log(0) call.
    y = replaceZeros(y, 1.0/pow(10, 100));
    y_pred = replaceZeros(y_pred, 1.0/pow(10, 100));

    if (y.size() != y_pred.size()) {
        throw std::invalid_argument("Error: Truth value and predicted value vectors should be the same size.");
    }

    double result = 0;

    for (int i = 0; i<y.size(); i++) {
        result -= y[i] * log2(y_pred[i]);
    }

    return result;
}

double Neuron::loss_function(std::vector<double> y, std::vector<double> y_pred) {
    return binary_cross_entropy(y, y_pred);
}



int main(int argc, char *argv[])
{
    Neuron n = Neuron(2, 0.1, std::vector<double>{0.5, 0.5, 0.5, 0.5 }); 

    double a = binary_cross_entropy(std::vector<double>{0, 0}, std::vector<double>{0, 0});

    std::cout << a << "\n";



    return 0;
}

