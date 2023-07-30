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
    double bias;
    std::vector<double> weights;
    double activation_function(double x);
    double combination_function(std::vector<double> weights, std::vector<double> inputs);
    double receive(std::vector<double> inputs);
    Neuron(double bias, std::vector<double> initial_weights) {
        this->weights = initial_weights;
        this->bias = bias;
    }
};

double Neuron::receive(std::vector<double> inputs) {
    double weighted_sum = this->combination_function(inputs, this->weights);
    double pred = std::round(this->activation_function(weighted_sum));

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
    if (y.size() != y_pred.size()) {
        throw std::invalid_argument("Error: Truth value and predicted value vectors should be the same size.");
    }

    double result = 0;

    for (int i = 0; i<y.size(); i++) {
        result -= y[i] * log2(y_pred[i]);
    }

    return result;
}

double cross_entropy(std::vector<double> p, std::vector<double> q) {
    // Remove zeros from arrays to avoid log(0) call.
    p = replaceZeros(p, 1.0/pow(10, 100));
    q = replaceZeros(q, 1.0/pow(10, 100));

    std::vector<double> logSum(p.size());
    double result = 0;

    // Calculate logs
    transform(p.begin(), p.end(), q.begin(), logSum.begin(), [](auto p, auto q) {return p * log2(q);});

    // Sum logs and negate
    result = -std::accumulate(logSum.begin(), logSum.end(), 0);

    return result;
}

int main(int argc, char *argv[])
{
    Neuron n = Neuron(2, std::vector<double>{0.5, 0.5, 0.5, 0.5 }); 

    double a = binary_cross_entropy(std::vector<double>{1, 0}, std::vector<double>{0.4, 0.6});

    std::cout << a << "\n";

    return 0;
}
