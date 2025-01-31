#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <bits/stdc++.h>
#include "Decoder.hpp"
#include "Types.hpp"
#include "MultilayerPerceptron.hpp"

vector<vector<double>> generateInitialLayerWeights(int layerDimension, int nextLayerDimension, double defaultValue, vector<int> ffnNetworkSpecs);

class Decoder
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

    Decoder(
        double learningRate,
        int headCount,
        int d_model,
        vector<int> ffnNetworkSpecs,
        vector<int> mmhaFfnNetworkSpecs,
        int heads);

    void mask(vector<vector<double>> &A);

    vector<vector<double>> scaledDotProductAttention(
        const vector<vector<double>> &K,
        const vector<vector<double>> &Q,
        const vector<vector<double>> &V,
        int dim,
        bool mask);

    vector<vector<double>> multiHeadAttention(const vector<double> &K, const vector<double> &Q, const vector<double> &V, bool mask);

    vector<double> addAndNorm(vector<double> v1, vector<double> v2);

    vector<double> forward(vector<double> inputs);

    void train(vector<double> x, vector<double> y);

    double lossFunction(vector<double> y, vector<double> y_pred);

    vector<double> softmax(vector<double> input);

    vector<vector<double>> softmax(vector<vector<double>> input);
};
