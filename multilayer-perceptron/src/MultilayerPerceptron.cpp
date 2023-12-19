
#include <iostream>
#include <vector>
#include <cmath>
#include "../../utils/utils.h"
#include "MultilayerPerceptron.h"


MultilayerPerceptron::MultilayerPerceptron(double initialBias, double initialWeightValue, int layerAmount, int layerDimension, double learningRate) {
    this->bias = initialBias;
    this->learningRate = learningRate;
    this->layerAmt = layerAmount;
    this->layerDim = layerDimension;

    // Initialize the weights with ones (np.ones(layer_amt, layer_d, layer_d))
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double> > weights_initial(this->layerDim, std::vector<double>(this->layerDim, initialWeightValue));
    for (int i = 0; i<this->layerAmt; i++) { 
        weights.push_back(weights_initial);
    }
}


double MultilayerPerceptron::combinationFunction(std::vector<double> weights, std::vector<double> inputs) {
    return weightedSum(weights, inputs) + this->bias;
}


double MultilayerPerceptron::activationFunction(double x) {
    return sigmoid(x);
}


double MultilayerPerceptron::forward(std::vector<std::vector<std::vector<double>>> inputs) {
    for (int i = 0; i<this->layerAmt; i++) {
        // for each hidden layer...
        this->combinationFunction(this->weights[i])
    }
}


void MultilayerPerceptron::train(std::vector<double> x, std::vector<double> y) {
    double yPred = forward(x);

    double eTotal = this->lossFunction(y, std::vector<double>{yPred});

    double* weightAdjustments = new double[weights.size()];

    // Now we must figure out for each weight, how much the weight contributed to the eTotal
    int i = 0;
    for (double weight : weights) {
        double eTotal_wrt_yPred = d_binary_cross_entropy(y, std::vector<double>{yPred});
        double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x) + bias);
        double weightedSum_wrt_weight = x[i];

        weightAdjustments[i] = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_weight;

        weights[i] -= this->learningRate * weightAdjustments[i];

        i += 1;
    }

    // same for bias term
    double eTotal_wrt_yPred = d_binary_cross_entropy(y, std::vector<double>{yPred});
    double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x));
    double weightedSum_wrt_bias = 1;
    double biasAdjustment = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_bias;
    bias -= learningRate * biasAdjustment;
}

double MultilayerPerceptron::lossFunction(std::vector<double> y, std::vector<double> yPred) {
    return binary_cross_entropy(y, yPred);
}
