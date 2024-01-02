
#include <iostream>
#include <vector>
#include <cmath>
#include "../../utils/utils.h"
#include "MultilayerPerceptron.h"
using std::vector;


MultilayerPerceptron::MultilayerPerceptron(double initialBias, double initialWeightValue, int hiddenLayerDim, int inputLayerDim, double outputLayerDim, double learningRate) {
    this->learningRate = learningRate;
    this->inputLayerDim = inputLayerDim;
    this->hiddenLayerDim = hiddenLayerDim;
    this->outputLayerDim = outputLayerDim;

    // Initialize the weights with ones (np.ones(layer_amt, layer_d, layer_d))
    vector<vector<vector<double>>> hiddenWeights;
    vector<vector<double>> hiddenWeightsInitial(hiddenLayerDim, vector<double>(hiddenLayerDim, initialWeightValue));
    for (int i = 0; i<this->hiddenWeights.size(); i++) {
        this->hiddenWeights.push_back(hiddenWeightsInitial);
    }
}


double MultilayerPerceptron::combinationFunction(vector<double> weights, vector<double> inputs) {
    return weightedSum(weights, inputs) + this->bias;
}


double MultilayerPerceptron::activationFunction(double x) {
    return sigmoid(x);
}


double MultilayerPerceptron::forward(vector<double> inputs) {
    // for each layer
    for (int i = 0; i<this->layerAmt; i++) {
        vector<vector<double>> weightsOfLayer = this->weights[i];

    }
}


void MultilayerPerceptron::train(vector<double> x, vector<double> y) {
    double yPred = forward(x);

    double eTotal = this->lossFunction(y, vector<double>{yPred});

    double* weightAdjustments = new double[weights.size()];

    // Now we must figure out for each weight, how much the weight contributed to the eTotal
    int i = 0;
    for (double weight : weights) {
        double eTotal_wrt_yPred = d_binary_cross_entropy(y, vector<double>{yPred});
        double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x) + bias);
        double weightedSum_wrt_weight = x[i];

        weightAdjustments[i] = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_weight;

        weights[i] -= this->learningRate * weightAdjustments[i];

        i += 1;
    }

    // same for bias term
    double eTotal_wrt_yPred = d_binary_cross_entropy(y, vector<double>{yPred});
    double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x));
    double weightedSum_wrt_bias = 1;
    double biasAdjustment = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_bias;
    bias -= learningRate * biasAdjustment;
}

double MultilayerPerceptron::lossFunction(vector<double> y, vector<double> yPred) {
    return binary_cross_entropy(y, yPred);
}
