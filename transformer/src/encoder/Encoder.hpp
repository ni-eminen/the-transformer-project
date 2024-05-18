#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <bits/stdc++.h>
#include "Encoder.hpp"
#include "Types.hpp"
#include "MultilayerPerceptron.hpp"

vector<vector<double>> generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue, vector<int> ffnNetworkSpecs);

class Encoder
{
public:
    double learningRate;
    int headCount;
    int d_model;
    int heads;
    vector<vector<MultilayerPerceptron>> qkvLinears;
    vector<int> ffnNetworkSpecs;
    vector<int> mmhaFfnNetworkSpecs;
    int d_k;
    int d_v;
    MultilayerPerceptron ffn;
    MultilayerPerceptron mmhaFfn;
    int blocks;

    Encoder(
        double learningRate,
        int headCount,
        int d_model,
        vector<int> ffnNetworkSpecs,
        vector<int> mmhaFfnNetworkSpecs,
        int heads);

    vector<vector<double>> scaledDotProductAttention(vector<vector<double>> K, vector<vector<double>> Q, vector<vector<double>> V, int dim);

    vector<double> multiHeadAttention(vector<double> K, vector<double> Q, vector<double> V);

    vector<double> addAndNorm(vector<double> v1, vector<double> v2);

    vector<double> forward(vector<double> inputs);

    void train(vector<double> x, vector<double> y);

    double lossFunction(vector<double> y, vector<double> y_pred);

    vector<double> softmax(vector<double> input);

    vector<vector<double>> softmax(vector<vector<double>> input);
};
