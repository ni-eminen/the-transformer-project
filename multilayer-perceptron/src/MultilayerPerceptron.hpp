#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>
#include "../../utils/Types.hpp"

matrix generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue);

class MultilayerPerceptron {
public:
    double learningRate;
    matrix inputWeights;
    matrix hiddenWeights;
    matrix outputWeights;
    matrix trainingBatchOutputs;
    matrix trainingBatchInputs;

    vector<double> hiddenBiases;
    vector<double> outputBiases;
    int hiddenLayerDim;
    int inputLayerDim;
    int outputLayerDim;
    int totalLayerAmt;

    vector<matrix> weights;
    matrix biases;

    MultilayerPerceptron(double initialBias, 
                         double initialWeightValue, 
                         int hiddenLayerDim, 
                         int inputLayerDim, 
                         int outputLayerDim, 
                         double learningRate);

    void print(std::string x);

    double combinationFunction(vector<double> weights, vector<double> inputs);

    double activationFunction(double x);

    vector<double> forward(vector<double> inputs);

    void train(vector<double> x, vector<double> y);

    double lossFunction(vector<double> y, vector<double> y_pred);
};

#endif // PERCEPTRON_H