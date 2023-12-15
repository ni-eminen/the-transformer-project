
#include <iostream>
#include <vector>
#include <cmath>
#include <math.h> 
#include "../../classes/NeuralNetwork.h"
#include "../../utils/utils.h"


class MultilayerPerceptron {
  public:
    double learningRate;
    double initial_bias;
    std::vector<double> weights;
    int layer_d;
    int layer_amt;

    MultilayerPerceptron(double initial_bias, int layerAmount, int layerDimension, double learningRate, double defaultWeightValue) {
        this->bias = initial_bias;
        this->learningRate = learningRate;

        std::vector<std::vector<std::vector<double>>> weights_init(layer_amt, defaultWeightValue);
        // for(int layer_i = 0; i < 20; i++) {
        //     layers[i] = weights_init<std::vector<double>>(layer_d);
        // }
    }


    double combinationFunction(std::std::vector<double> weights, std::std::vector<double> inputs) {
        return weightedSum(weights, inputs) + this->bias;
    }


    double activationFunction(double x) {
        return sigmoid(x);
    }


    double forward(std::std::vector<double> inputs) {
        double weightedSum = combinationFunction(inputs, weights);
        double activationOutput = activationFunction(weightedSum);

        return activationOutput;
    }


    void train(std::std::vector<double> x, std::std::vector<double> y) {
        double yPred = forward(x);

        double eTotal = lossFunction(y, std::std::vector<double>{yPred});

        double* weightAdjustments = new double[weights.size()];

        // Now we must figure out for each weight, how much the weight contributed to the eTotal
        int i = 0;
        for (double weight : weights) {
            double eTotal_wrt_yPred = d_binary_cross_entropy(y, std::std::vector<double>{yPred});
            double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x) + bias);
            double weightedSum_wrt_weight = x[i];

            weightAdjustments[i] = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_weight;

            weights[i] -= learningRate * weightAdjustments[i];

            i += 1;
        }

        // same for bias term
        double eTotal_wrt_yPred = d_binary_cross_entropy(y, std::std::vector<double>{yPred});
        double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x));
        double weightedSum_wrt_bias = 1;
        double biasAdjustment = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_bias;
        bias -= learningRate * biasAdjustment;
    }

    double lossFunction(std::std::vector<double> y, std::std::vector<double> yPred) {
        return binary_cross_entropy(y, yPred);
    }
};