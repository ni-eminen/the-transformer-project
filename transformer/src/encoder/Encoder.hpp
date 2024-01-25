#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include "Types.hpp"

vector<vector<double>> generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue);

class Encoder
{
public:
    double learningRate;

    Encoder(vector<int> networkSpecs,
            double initialBias,
            double initialWeightValue,
            double learningRate,
            int heads,
            int d_model);

    vector<double> scaledDotProductAttention(vector<vector<double>> K, vector<vector<double>> Q, vector<vector<double>> V, int dim);

    double multiHeadAttention(vector<double> K, vector<double> Q, vector<double> V);

    double maskedMultiHeadAttention(vector<double> K, vector<double> Q, vector<double> V);

    vector<double> addAndNorm(vector<vector<double>> V);

    vector<double> forward(vector<double> inputs);

    void train(vector<double> x, vector<double> y);

    double lossFunction(vector<double> y, vector<double> y_pred);

    vector<double> softmax(vector<double> input);

    vector<vector<double>> softmax(vector<vector<double>> input);
};
