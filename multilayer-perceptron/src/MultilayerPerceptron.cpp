
#include <iostream>
#include <vector>
#include <cmath>
#include "../../utils/utils.h"
#include "MultilayerPerceptron.h"
using std::vector;

vector<vector<double>> generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue = 1) {
    vector<vector<double>> weightsInitial(layerDimension, vector<double>(layerDimension, defaultValue));
    return weightsInitial;
}

MultilayerPerceptron::MultilayerPerceptron(double initialBias, double initialWeightValue, int hiddenLayerDim, int inputLayerDim, double outputLayerDim, double learningRate) {
    this->learningRate = learningRate;
    this->inputLayerDim = inputLayerDim;
    this->hiddenLayerDim = hiddenLayerDim;
    this->outputLayerDim = outputLayerDim;

    vector<vector<double>> inputWeights(inputLayerDim, generateInitialLayerWeights(hiddenLayerDim, hiddenLayerDim, initialWeightValue))
    vector<vector<vector<double>>> hiddenWeights(hiddenLayerDim-1, generateInitialLayerWeights(hiddenLayerDim, hiddenLayerDim, initialWeightValue));
    hiddenWeights.push_back(generateInitialLayerWeights(hiddenLayerDim, outputLayerDim))
}


double MultilayerPerceptron::combinationFunction(vector<double> weights, vector<double> inputs) {
    return weightedSum(weights, inputs) + this->bias;
}


double MultilayerPerceptron::activationFunction(double x) {
    return sigmoid(x);
}


double MultilayerPerceptron::forward(vector<double> inputs) {

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
