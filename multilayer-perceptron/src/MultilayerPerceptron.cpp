
#include <iostream>
#include <vector>
#include <cmath>
#include "../../utils/LinearAlgebra.hpp"
#include "../../utils/utils.hpp"
#include "MultilayerPerceptron.hpp"
using std::vector;
using matrix = std::vector<std::vector<double>>;

vector<vector<double>> generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue = 1) {
    vector<vector<double>> weightsInitial(layerDimension, vector<double>(nextLayerDimension, defaultValue));
    return weightsInitial;
}

MultilayerPerceptron::MultilayerPerceptron(double initialBias, double initialWeightValue, int hiddenLayerDim, int inputLayerDim, int outputLayerDim, double learningRate) {
    this->learningRate = learningRate;
    this->inputLayerDim = inputLayerDim;
    this->hiddenLayerDim = hiddenLayerDim;
    this->outputLayerDim = outputLayerDim;

    vector<double> hiddenBiases(hiddenLayerDim, initialBias);
    this->hiddenBiases = hiddenBiases;
    vector<double> outputBiases(outputLayerDim, initialBias);
    this->outputBiases = outputBiases;

    this->inputWeights = generateInitialLayerWeights(inputLayerDim, hiddenLayerDim, initialWeightValue);
    auto hiddenWeights = generateInitialLayerWeights(hiddenLayerDim, outputLayerDim, initialWeightValue);
    this->hiddenWeights = hiddenWeights;

    // All weights in one 3d vector
    vector<vector<vector<double>>> weights;
    weights.push_back(this->inputWeights);
    weights.push_back(this->hiddenWeights);
    this->weights = weights;

    // All biases in one 2d vector
    vector<vector<double>> biases;
    biases.push_back(this->hiddenBiases);
    biases.push_back(this->outputBiases);
    this->biases = biases;
}


double MultilayerPerceptron::activationFunction(double x) {
    return sigmoid(x);
}


vector<double> MultilayerPerceptron::forward(vector<double> inputs) {
    vector<double> outputs = inputs;
    double weightedSumsPlusBias;
    for (int layer_i = 0; layer_i<this->weights.size(); layer_i++) {
        vector<double> weightedSums = matMul(outputs, this->weights[layer_i]);
        outputs = weightedSums;

        for (int neuron_i = 0; neuron_i<weightedSums.size(); neuron_i++) {
            weightedSumsPlusBias = weightedSums[neuron_i] + this->biases[layer_i][neuron_i];
            outputs[neuron_i] = this->activationFunction(weightedSumsPlusBias);
        }
    }

    return outputs;
}


void MultilayerPerceptron::train(vector<double> x, vector<double> y) {
    // double yPred = forward(x);

    // double eTotal = this->lossFunction(y, vector<double>{yPred});

    // double* weightAdjustments = new double[weights.size()];

    // // Now we must figure out for each weight, how much the weight contributed to the eTotal
    // int i = 0;
    // for (double weight : weights) {
    //     double eTotal_wrt_yPred = d_binary_cross_entropy(y, vector<double>{yPred});
    //     double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x) + bias);
    //     double weightedSum_wrt_weight = x[i];

    //     weightAdjustments[i] = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_weight;

    //     weights[i] -= this->learningRate * weightAdjustments[i];

    //     i += 1;
    // }

    // // same for bias term
    // double eTotal_wrt_yPred = d_binary_cross_entropy(y, vector<double>{yPred});
    // double yPred_wrt_weightedSum = dSigmoid(combinationFunction(weights, x));
    // double weightedSum_wrt_bias = 1;
    // double biasAdjustment = eTotal_wrt_yPred * yPred_wrt_weightedSum * weightedSum_wrt_bias;
    // bias -= learningRate * biasAdjustment;
}

double MultilayerPerceptron::lossFunction(vector<double> y, vector<double> yPred) {
    // return binary_cross_entropy(y, yPred);
    return 1.0;
}
