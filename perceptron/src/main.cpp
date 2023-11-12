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
#include <string>

class Neuron {
  public:
    double learning_rate;
    double bias;
    std::vector<double> weights;
    double predict(std::vector<double> inputs);
    double loss_function(std::vector<double> y, std::vector<double> y_pred);
    Neuron(double bias, double learning_rate, std::vector<double> initial_weights) {
        this->weights = initial_weights;
        this->bias = bias;
        this->learning_rate = learning_rate;
    }

    void print(std::string x) {
        std::cout << x << std::endl;
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

    double combination_function(std::vector<double> weights, std::vector<double> inputs) {
        return weighted_sum(weights, inputs);
    }

    double activation_function(double x) {
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

    double cross_entropy(std::vector<double> y, std::vector<double> y_pred) {
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

        result = (1/y.size()) * result; 

        return result;
    }


    double d_cross_entropy(std::vector<double> y, std::vector<double> y_pred) {
        // Remove zeros from arrays to avoid log(0) call.
        y = replaceZeros(y, 1.0/pow(10, 100));
        y_pred = replaceZeros(y_pred, 1.0/pow(10, 100));

        if (y.size() != y_pred.size()) {
            throw std::invalid_argument("Error: Truth value and predicted value vectors should be the same size.");
        }

        double result = 0;

        for (int i = 0; i<y.size(); i++) {
            result -= y[i] * (1 / y_pred[i] * log(2));

            std::cout << "resutl:" << result << std::endl;
        }

        result = (1/y.size()) * result; 

        return result;
    }

    std::vector<double> forward_propagate(std::vector<double> x) {
        return std::vector<double>{0.5, 0.5};
    }


    void train(std::vector<double> x, std::vector<double> y) {
        std::vector<double> y_pred = Neuron::forward_propagate(x);

        for (double x : y_pred) {
            std::cout << x;
        }

        for (double x : y) {
            std::cout << x;
        }

        double E_total = cross_entropy(y_pred, y);

        std::cout << "E_total: " << E_total;
        std::cout << "y_pred: " << std::string(y_pred.begin(), y_pred.end());

        double* weight_adjustments = new double[weights.size()];

        // now we must figure out for each weight, how much the weight contributed to the E_total
        int i = 0;
        for (double weight : weights) {
            double E_total_wrt_y_pred = d_cross_entropy(y, y_pred);
            double y_pred_wrt_weighted_sum = d_sigmoid(weighted_sum(weights, x));
            double weighted_sum_wrt_weight = x[i];

            weight_adjustments[i] = E_total_wrt_y_pred * y_pred_wrt_weighted_sum * weighted_sum_wrt_weight;

            weights[i] -= learning_rate * weight_adjustments[i];

            i += 1;
        }
    }
};

double Neuron::predict(std::vector<double> inputs) {
    double weighted_sum = this->combination_function(inputs, this->weights);
    double activation_output = this->activation_function(weighted_sum);
    double pred = std::round(activation_output);

    return pred;
}


double Neuron::loss_function(std::vector<double> y, std::vector<double> y_pred) {
    return cross_entropy(y, y_pred);
}



int main(int argc, char *argv[])
{
    Neuron perceptron = Neuron(1, 0.5, std::vector<double>{0.5, 0.5, 0.5, 0.5 });

    std::vector<double> x = std::vector<double>{1, 0, 0, 1};
    std::vector<double> y = std::vector<double>{1, 0};

    perceptron.train(x, y);

    return 0;
}

