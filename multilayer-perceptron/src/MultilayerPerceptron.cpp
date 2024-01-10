
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include "LinearAlgebra.hpp"
#include "utils.hpp"
#include "MultilayerPerceptron.hpp"
#include "Types.hpp"
#include "stdlib.h"

vector<vector<double> > generateInitialLayerWeights(int layerDimension, int nextLayerDimension)
{
    vector<vector<double> > weightsInitial(layerDimension, vector<double>());
    for (int j = 0; j < weightsInitial.size(); j++)
    {
        for (int i = 0; i < nextLayerDimension; i++)
        {
            double r = ((double)rand() / (RAND_MAX));
            weightsInitial[j].push_back(r);
        }
    }
    return weightsInitial;
}

MultilayerPerceptron::MultilayerPerceptron(vector<int> networkSpecs, double initialBias, double initialWeightValue, double learningRate)
{
    this->learningRate = learningRate;
    this->inputLayerDim = networkSpecs[0];
    this->hiddenLayerDim = networkSpecs[1];
    this->outputLayerDim = networkSpecs[networkSpecs.size() - 1];
    this->hiddenLayerAmount = networkSpecs.size() - 2;

    vector<vector<double> > hiddenBiases(hiddenLayerAmount, vector<double>(hiddenLayerDim, initialBias));
    vector<double> outputBiases(outputLayerDim, initialBias);
    this->outputBiases = outputBiases;

    this->inputWeights = generateInitialLayerWeights(inputLayerDim, hiddenLayerDim);
    vector<vector<vector<double> > > hiddenWeights;
    for (int i = 0; i < hiddenLayerAmount; i++)
        hiddenWeights.push_back(generateInitialLayerWeights(networkSpecs[i + 1], networkSpecs[i + 2]));

    // All weights in one 3d vector
    vector<vector<vector<double> > > weights;
    weights.push_back(this->inputWeights);
    for (int i = 0; i < hiddenWeights.size(); i++)
        weights.push_back(hiddenWeights[i]);
    this->weights = weights;

    // All biases in one 2d vector
    vector<vector<double> > biases;
    biases.push_back(this->outputBiases);
    for (int i = 0; i < hiddenBiases.size(); i++)
    {
        biases.push_back(hiddenBiases[i]);
    }
    this->biases = biases;

    this->totalLayerAmt = this->weights.size();
}

double MultilayerPerceptron::activation(double x)
{
    return sigmoid(x);
}
double MultilayerPerceptron::dActivation(double x)
{
    return dSigmoid(x);
}

double MultilayerPerceptron::outputActivation(double x)
{
    return sigmoid(x);
}
double MultilayerPerceptron::dOutputActivation(double x)
{
    return dSigmoid(x);
}

double MultilayerPerceptron::loss(std::vector<double> y, std::vector<double> y_pred)
{
    return binary_cross_entropy(y, y_pred);
}
double MultilayerPerceptron::dLoss(std::vector<double> y, std::vector<double> y_pred)
{
    return d_binary_cross_entropy(y, y_pred);
}

vector<double> MultilayerPerceptron::forward(vector<double> inputs)
{
    if (inputs.size() != this->inputLayerDim)
    {
        throw std::invalid_argument("Input vector must be same length as input layer dimension.");
    }

    vector<vector<double> > forwardIns(0);
    vector<vector<double> > forwardOuts(0);
    this->forwardIns = forwardIns;
    this->forwardOuts = forwardOuts;

    this->forwardIns.push_back(inputs);
    this->forwardOuts.push_back(inputs);

    // vector<double> outputs = inputs;
    vector<double> outputs;
    std::copy(inputs.begin(), inputs.end(), std::back_inserter(outputs));
    double weightedSumsPlusBias;
    for (int layer_i = 0; layer_i < this->weights.size(); layer_i++)
    {
        vector<double> weightedSums = matMul(outputs, this->weights[layer_i])[0];
        outputs = vector<double>();
        vector<double> weightedSumPlusBias = elementWiseSum(weightedSums, this->biases[layer_i]);

        for (int neuron_i = 0; neuron_i < weightedSums.size(); neuron_i++)
        {
            // output layer
            if (layer_i == this->weights.size() - 1)
            {
                outputs.push_back(this->outputActivation(weightedSumPlusBias[neuron_i]));
            }
            else
            {
                outputs.push_back(this->activation(weightedSumPlusBias[neuron_i]));
            }
        }
        // Saving these for backpropagation
        this->forwardIns.push_back(weightedSumPlusBias);
        this->forwardOuts.push_back(outputs);
    }

    return outputs;
}

void MultilayerPerceptron::train(vector<double> x, vector<double> y)
{
    vector<double> output = this->forward(x);
    double eTotal = this->loss(y, output);
    double eTotalWrtPrediction = this->dLoss(y, output);
    vector<vector<vector<double> > > eTotalWrtWeight(this->weights.size(), vector<vector<double> >(hiddenLayerDim, vector<double>()));
    vector<vector<double> > star = axbVector(this->totalLayerAmt, this->hiddenLayerDim);
    vector<vector<double> > starBiases = axbVector(this->totalLayerAmt, this->hiddenLayerDim);
    vector<vector<double> > eTotalWrtBias(this->weights.size(), vector<double>());

    // Traverse the layers backwards starting from second to last layer
    for (int layer_i = this->weights.size() - 1; layer_i >= 0; layer_i--)
    {
        // Traverse the neurons on layer_i
        for (int neuron_i = 0; neuron_i < weights[layer_i].size(); neuron_i++)
        {
            // Traverse the weights of neuron_i
            double eTotalWrtOutput;
            double outputWrtWeightedSum;
            for (int weight_i = 0; weight_i < this->forwardOuts[layer_i + 1].size(); weight_i++)
            {
                if (layer_i == weights.size() - 1)
                {
                    eTotalWrtOutput = dLoss(y, this->forwardOuts[layer_i + 1]);
                }
                else
                {
                    // o is used as iterator variable as it represenths the oth output neuron.
                    // o is commonly used to denote the output layer's neurons
                    for (int o = 0; o < this->weights[layer_i + 1].size(); o++)
                    {
                        eTotalWrtOutput += star[layer_i + 1][o] * this->weights[layer_i + 1][weight_i][o];
                    }
                }

                outputWrtWeightedSum = this->dActivation(this->forwardIns[layer_i + 1][weight_i]);
                double weightedSumWrtWeight = this->forwardOuts[layer_i][neuron_i];

                eTotalWrtWeight[layer_i][neuron_i].push_back(eTotalWrtOutput * outputWrtWeightedSum * weightedSumWrtWeight);
                star[layer_i][neuron_i] += eTotalWrtOutput * outputWrtWeightedSum;
            }
            // Bias
            if (layer_i == weights.size() - 1)
            {
                eTotalWrtOutput = dLoss(y, this->forwardOuts[layer_i + 1]);
            }
            else
            {
                for (int o = 0; o < this->weights[layer_i + 1].size(); o++)
                {
                    eTotalWrtOutput += starBiases[layer_i + 1][o] * this->weights[layer_i + 1][neuron_i][o];
                }
            }
            outputWrtWeightedSum = dActivation(this->forwardIns[layer_i + 1][neuron_i]);
            eTotalWrtBias[layer_i].push_back(eTotalWrtOutput * outputWrtWeightedSum);
            starBiases[layer_i][neuron_i] = eTotalWrtOutput * outputWrtWeightedSum;
        }
    }

    for (int layer_i = 0; layer_i < this->weights.size(); layer_i++)
    {
        for (int neuron_i = 0; neuron_i < this->weights[layer_i].size(); neuron_i++)
        {
            for (int weight_i = 0; weight_i < this->weights[layer_i][neuron_i].size(); weight_i++)
            {
                this->weights[layer_i][neuron_i][weight_i] -= this->learningRate * eTotalWrtWeight[layer_i][neuron_i][weight_i];
            }
            this->biases[layer_i][neuron_i] -= this->learningRate * eTotalWrtBias[layer_i][neuron_i];
        }
    }
}
