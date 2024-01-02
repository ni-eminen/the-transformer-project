
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
    printMatrix(hiddenWeights, "hidden weight after init");
}


double MultilayerPerceptron::activationFunction(double x) {
    return sigmoid(x);
}


vector<double> MultilayerPerceptron::forward(vector<double> inputs) {
    // this->inputWeights * inputs_1 = [i_1, ..., i_n] + biases = sum => activation(sum) = inputs_2 => repeat with hidden
    vector<double> weightedSums = matMul(inputs, this->inputWeights);
    printVector(weightedSums, "weighted sums without bias");


    vector<double> activated = weightedSums;
    double weightedSumPlusBias;
    for (int i = 0; i<weightedSums.size(); i++) {
        weightedSumPlusBias = weightedSums[i] + this->hiddenBiases[i];
        activated[i] = this->activationFunction(weightedSumPlusBias);
    }

    printVector(activated, "Hidden layer output");

    printMatrix(this->hiddenWeights, "hidden weights");
    vector<double> weightedSums_2 = matMul(activated, this->hiddenWeights);

    printVector(weightedSums_2, "2");

    activated = weightedSums_2;
    for (int i = 0; i<weightedSums_2.size(); i++) {
        weightedSumPlusBias = weightedSums[i] + this->outputBiases[i];
        activated[i] = this->activationFunction(weightedSumPlusBias);
    }

    return activated;
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
