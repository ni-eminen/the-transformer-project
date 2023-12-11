#include <iostream>
#include <vector>
#include <cmath>
#include <math.h> 
#include "../../utils/utils.h"


class Perceptron {
  public:
    double learning_rate;
    double bias;
    std::vector<double> weights;
    Perceptron(double bias, double learning_rate, std::vector<double> initial_weights) {
        this->weights = initial_weights;
        this->bias = bias;
        this->learning_rate = learning_rate;
    }

    double weightedSum(std::vector<double> weights, std::vector<double> inputs) {
        double result = 0;

        for(int i = 0; i<weights.size(); i++) {
            result += (weights[i] * inputs[i]);
        }

        return result + bias;
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    double dSigmoid(double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    double combinationFunction(std::vector<double> weights, std::vector<double> inputs) {
        return weightedSum(weights, inputs);
    }

    double activationFunction(double x) {
        return sigmoid(x);
    }

    double forwardPropagate(std::vector<double> inputs) {
        double weightedSum = combinationFunction(inputs, weights);
        double activation_output = activationFunction(weightedSum);

        return activation_output;
    }

    void train(std::vector<double> x, std::vector<double> y) {
        double y_pred = forwardPropagate(x);

        double E_total = lossFunction(y, std::vector<double>{y_pred});

        double* weight_adjustments = new double[weights.size()];

        // Now we must figure out for each weight, how much the weight contributed to the E_total
        int i = 0;
        for (double weight : weights) {
            double E_total_wrt_y_pred = d_binary_cross_entropy(y, std::vector<double>{y_pred});
            double y_pred_wrt_weightedSum = dSigmoid(weightedSum(weights, x));
            double weightedSum_wrt_weight = x[i];

            weight_adjustments[i] = E_total_wrt_y_pred * y_pred_wrt_weightedSum * weightedSum_wrt_weight;

            weights[i] -= learning_rate * weight_adjustments[i];

            i += 1;
        }

        // same for bias term
        double E_total_wrt_y_pred = d_binary_cross_entropy(y, std::vector<double>{y_pred});
        double y_pred_wrt_weightedSum = dSigmoid(weightedSum(weights, x));
        double weightedSum_wrt_bias = 1;
        double biasAdjustment = E_total_wrt_y_pred * y_pred_wrt_weightedSum * weightedSum_wrt_bias;
        bias -= learning_rate * biasAdjustment;
    }

    double lossFunction(std::vector<double> y, std::vector<double> y_pred) {
        return binary_cross_entropy(y, y_pred);
    }
};