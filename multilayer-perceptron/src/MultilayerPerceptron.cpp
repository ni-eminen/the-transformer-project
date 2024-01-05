
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

    this->totalLayerAmt = this->weights.size();
}


double MultilayerPerceptron::activationFunction(double x) {
    return sigmoid(x);
}


vector<double> MultilayerPerceptron::forward(vector<double> inputs) {
    matrix trainingBatchInputs(this->weights.size());
    matrix trainingBatchOutputs(this->weights.size());
    this->trainingBatchInputs = trainingBatchInputs;
    this->trainingBatchOutputs = trainingBatchOutputs;

    vector<double> outputs = inputs;
    double weightedSumsPlusBias;
    for (int layer_i = 0; layer_i<this->weights.size(); layer_i++) {
        vector<double> weightedSums = matMul(outputs, this->weights[layer_i]);

        vector weightedSumPlusBias = elementWiseSum(weightedSums, this->biases[layer_i]);

        for (int neuron_i = 0; neuron_i<weightedSums.size(); neuron_i++) {
            outputs[neuron_i] = this->activationFunction(weightedSumPlusBias[neuron_i]);
        }
        // Saving these for backpropagation
        this->trainingBatchInputs.push_back(weightedSumPlusBias);
        this->trainingBatchOutputs.push_back(outputs);
    }

    return outputs;
}


void MultilayerPerceptron::train(vector<double> x, vector<double> y) {
    vector<double> output = this->forward(x);
    double eTotal = lossFunction(y, output);
    double eTotalWrtPrediction = d_binary_cross_entropy(y, output);
    vector<vector<vector<double>>> eTotalWrtWeight(this->weights.size(), vector<vector<double>>(1, vector<double>{}));
    matrix eTotalWrtBias(this->weights.size(), vector<double>{});
    // Traverse the layers backwards starting from second to last layer
    for (int layer_i = this->weights.size()-2; layer_i>=0; layer_i--) {
        // Traverse the neurons on layer_i
        for (int neuron_i = 0; neuron_i<weights[layer_i].size(); neuron_i++) {
            // Traverse the weights of neuron_i
            double eTotalWrtOutput;
            double outputWrtWeightedSum;
            for (int weight_i = 0; weight_i<weights[layer_i+1].size(); weight_i++) {
                if (layer_i == weights.size() - 2) {
                    eTotalWrtOutput = d_binary_cross_entropy(y, vector<double>{this->trainingBatchOutputs[layer_i+1][weight_i]});
                } else {
                    // o is used as iterator variable as it represenths the oth output neuron. 
                    // o is commonly used to denote the output layer's neurons
                    for (int o = 0; o<this->outputLayerDim; o++) {
                        eTotalWrtOutput += d_binary_cross_entropy(y, vector<double>{this->trainingBatchOutputs[this->totalLayerAmt][o]}) * dSigmoid(this->trainingBatchInputs[this->totalLayerAmt][o]) * this->weights[this->totalLayerAmt-2][weight_i][o];
                    }
                }
                // matrix outputWrtWeightedSum(this->weights.size(), vector<double>{});
                // matrix weightedSumWrtWeight(this->weights.size(), vector<double>{});
                // outputWrtWeightedSum[layer_i].push_back(dSigmoid(this->trainingBatchInputs[layer_i+1][weight_i]));
                // weightedSumWrtWeight[layer_i].push_back(this-weights[layer_i][neuron_i][weight_i]);

                outputWrtWeightedSum = dSigmoid(this->trainingBatchInputs[layer_i+1][weight_i]);
                double weightedSumWrtWeight = this->weights[layer_i][neuron_i][weight_i];
                
                eTotalWrtWeight[layer_i][neuron_i].push_back(eTotalWrtOutput * outputWrtWeightedSum * weightedSumWrtWeight);
            }
            // same for bias term
            // here we need to save each neuron's output to give to dSigmoid
            // double weightedSum_wrt_bias = 1;
            // double biasAdjustment = eTotalWrtOutput * outputWrtWeightedSum * weightedSum_wrt_bias;
            // eTotalWrtBias[layer_i].push_back(biasAdjustment);
        }
    }
}

double MultilayerPerceptron::lossFunction(vector<double> y, vector<double> yPred) {
    return binary_cross_entropy(y, yPred);
}
