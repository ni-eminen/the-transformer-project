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
    vector<vector<vector<double> > > hiddenWeights;
    vector<vector<double> > outputWeights;
    vector<vector<double> > forwardOuts;
    vector<vector<double> > forwardIns;

    vector<vector<double> > hiddenBiases;
    vector<double> outputBiases;
    int hiddenLayerDim;
    int inputLayerDim;
    int outputLayerDim;
    int totalLayerAmt;
    int hiddenLayerAmount;
    vector<vector<vector<double> > > weights;
    vector<vector<double> > biases;

    MultilayerPerceptron(vector<int> networkSpecs,
                         double initialBias,
                         double initialWeightValue,
                         double learningRate);

    void print(std::string x);

    double combinationFunction(vector<double> weights, vector<double> inputs);

    double activation(double x);
    double dActivation(double x);
    double outputActivation(double x);
    double dOutputActivation(double x);
    double loss(std::vector<double> y, std::vector<double> y_pred);
    double dLoss(std::vector<double> y, std::vector<double> y_pred);

    vector<double> forward(vector<double> inputs);

    void train(vector<double> x, vector<double> y);

    double lossFunction(vector<double> y, vector<double> y_pred);
};

#endif // PERCEPTRON_H