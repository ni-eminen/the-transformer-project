
#include <iostream>
#include <vector>
#include <cmath>
#include "../../utils/LinearAlgebra.hpp"
#include "../../utils/utils.hpp"
#include "MultilayerPerceptron.hpp"
#include "../../utils/Types.hpp"


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
    vector<double> output = this->forward(x);
    double eTotal = lossFunction(y, output);
    double eTotalWrtPrediction = d_binary_cross_entropy(y, output);
    matrix eTotalWrtWeight(this->weights.size(), vector<double>{});
    matrix eTotalWrtBias(this->weights.size(), vector<double>{});
    // Traverse the layers backwards
    for (int layer_i = this->weights.size()-1; layer_i>=0; layer_i++) {
        // Traverse the neurons on layer_i
        for (int neuron_i = 0; neuron_i<weights[layer_i].size(); neuron_i++) {
            // Traverse the weights of neuron_i
            for (int weight_i = 0; weight_i<weights[layer_i+1].size(); weight_i++) {
                double eTotalWrtOutput;
                if (layer_i == this->weights.size() - 1) {
                    eTotalWrtOutput = eTotalWrtPrediction;
                } else {

                    eTotalWrtOutput = eTotalWrtWeight[layer_i + 1][weight_i];
                }
                matrix outputWrtWeightedSum(this->weights.size(), vector<double>{});
                matrix weightedSumWrtWeight(this->weights.size(), vector<double>{});
                double outputWrtWeightedSum = dSigmoid(combinationFunction(weights, x));
                double weightedSumWrtWeight = outputs[layer_i][neuron_i][weight_i];
                eTotalWrtWeight[layer_i][neuron_i].push_back(eTotalWrtOutput * outputWrtWeightedSum * weightedSumWrtWeight)
            }
            // same for bias term
            double eTotalWrtPrediction = d_binary_cross_entropy(y, output);
            double outputWrtWeightedSum = dSigmoid(combinationFunction(weights, x));
            double weightedSum_wrt_bias = 1;
            double biasAdjustment = eTotalWrtPrediction * outputWrtWeightedSum * weightedSum_wrt_bias;
            bias -= learningRate * biasAdjustment;
            eTotalWrtBias[layer_i].push_back(biasAdjustment);
        }

    }
}

double MultilayerPerceptron::lossFunction(vector<double> y, vector<double> yPred) {
    return binary_cross_entropy(y, yPred);
}
