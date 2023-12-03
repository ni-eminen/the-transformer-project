using namespace std;
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
#include "utils.h"

class Neuron {
  public:
    double learning_rate;
    double bias;
    std::vector<double> weights;
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

        return result + bias;
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

    double binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred) {
        y = replaceZeros(y, 1.0/pow(10, 100));
        y_pred = replaceZeros(y_pred, 1.0/pow(10, 100));

        double result = 0;
        for(int i = 0; i<y.size(); i++) {
            double temp = y[i] * log(y_pred[i]); // + (1-y[i]) * log(1-y_pred[i]);
            result = result + temp;
        }   

        return -result;
    }



    double d_binary_cross_entropy(std::vector<double> y, std::vector<double> y_pred) {
        y = replaceZeros(y, 1.0/pow(10, 100));
        y_pred = replaceZeros(y_pred, 1.0/pow(10, 100));

        double result = 0;
        for(int i = 0; i<y.size(); i++) {
            double temp = log(y_pred[i]);//(y[i]-y_pred[i]) / (y_pred[i]*(-y_pred[i] + 1));
            result = result + temp;
        }

        return -result;
    }

    double forward_propagate(std::vector<double> inputs) {
        double weighted_sum = combination_function(inputs, weights);
        double activation_output = activation_function(weighted_sum);

        return activation_output;
    }

    void train(std::vector<double> x, std::vector<double> y) {
        double y_pred = forward_propagate(x);

        double E_total = loss_function(y, std::vector<double>{y_pred});

        double* weight_adjustments = new double[weights.size()];

        // Now we must figure out for each weight, how much the weight contributed to the E_total
        int i = 0;
        for (double weight : weights) {
            double E_total_wrt_y_pred = d_binary_cross_entropy(y, std::vector<double>{y_pred});
            double y_pred_wrt_weighted_sum = d_sigmoid(weighted_sum(weights, x));
            double weighted_sum_wrt_weight = x[i];

            weight_adjustments[i] = E_total_wrt_y_pred * y_pred_wrt_weighted_sum * weighted_sum_wrt_weight;

            weights[i] -= learning_rate * weight_adjustments[i];

            i += 1;
        }

        // same for bias term
        double E_total_wrt_y_pred = d_binary_cross_entropy(y, std::vector<double>{y_pred});
        double y_pred_wrt_weighted_sum = d_sigmoid(weighted_sum(weights, x));
        double weighted_sum_wrt_bias = this->bias;
        double biasAdjustment = E_total_wrt_y_pred * y_pred_wrt_weighted_sum * weighted_sum_wrt_bias;

        this->bias = learning_rate * biasAdjustment;
    }

    double loss_function(std::vector<double> y, std::vector<double> y_pred) {
        return binary_cross_entropy(y, y_pred);
    }
};


int main(int argc, char *argv[])
{
    // We want the perceptron to give and-gate logic:
    // input :  output
    // 1 1      1
    // 1 0      0
    // 0 1      0
    // 0 0      0
    Neuron perceptron = Neuron(-1, .1, std::vector<double>{-1, 1 });

    std::vector<std::vector<double>> X = std::vector<std::vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    std::vector<std::vector<double>> y = std::vector<std::vector<double>>{{1},      {0},    {0},    {0}};

    std::vector<double> prev_w = perceptron.weights;

    // Training
    for(int i=0;i<1000000;i++) {
        for(int i=0;i<X.size();i++) perceptron.train(X[i], y[i]);
    }

    double pred = perceptron.forward_propagate(std::vector<double> {1, 1});
    std::cout << pred << std::endl;

    return 0;
}

