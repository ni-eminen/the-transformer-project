#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <iostream>
#include <vector>
#include <cmath>
#include "Types.hpp"

vector<vector<double> > generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue);

class MultilayerPerceptron
{
public:
    double learningRate;
    vector<vector<double> > inputWeights;
    vector<vector<double> > hiddenWeights;
    vector<vector<double> > outputWeights;
    vector<vector<double> > forwardOuts;
    vector<vector<double> > forwardIns;

    vector<double> hiddenBiases;
    vector<double> outputBiases;
    int hiddenLayerDim;
    int inputLayerDim;
    int outputLayerDim;
    int totalLayerAmt;

    vector<vector<vector<double> > > weights;
    vector<vector<double> > biases;

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