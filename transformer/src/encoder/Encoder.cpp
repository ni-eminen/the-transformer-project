using namespace std;
#include <string>
#include <bits/stdc++.h>
#include "Encoder.hpp"
#include "Types.hpp"
#include "utils.hpp"
#include "LinearAlgebra.hpp"
#include "MultilayerPerceptron.hpp"

Encoder::Encoder(
    double learningRate,
    int headCount,
    int d_model,
    vector<int> ffnNetworkSpecs,
    vector<int> mmhaFfnNetworkSpecs,
    int heads) : learningRate(learningRate),
                 headCount(headCount),
                 d_model(d_model),
                 ffnNetworkSpecs(ffnNetworkSpecs),
                 mmhaFfnNetworkSpecs(mmhaFfnNetworkSpecs),
                 blocks(5),
                 ffn(ffnNetworkSpecs, 1, 0.5, learningRate, &sigmoid, &dSigmoid),
                 mmhaFfn(mmhaFfnNetworkSpecs, 1, 0.5, learningRate, &linear, &dLinear)
{
  // Hyperparams
  this->heads = heads;
  this->learningRate = learningRate;
  this->headCount = headCount;
  this->d_model = d_model;
  this->ffnNetworkSpecs = ffnNetworkSpecs;
  this->mmhaFfnNetworkSpecs = mmhaFfnNetworkSpecs;
  this->blocks = 5;
  int initialBias = 0;
  double initialWeightValue = 0.5;

  // Feed-Forward layer
  // TODO: Give activation function to Encoder as argument
  MultilayerPerceptron ffn = MultilayerPerceptron(ffnNetworkSpecs, 1, .5, this->learningRate, &sigmoid, &dSigmoid);
  this->ffn = ffn;
  MultilayerPerceptron mmhaFfn = MultilayerPerceptron(mmhaFfnNetworkSpecs, 1, .5, this->learningRate, &linear, &dLinear);
  this->mmhaFfn = mmhaFfn;

  // Linear projectors for multihead-attention
  vector<vector<MultilayerPerceptron>> qkvLinears = vector<vector<MultilayerPerceptron>>(3, vector<MultilayerPerceptron>());
  for (int i = 0; i < 3; i++)
  {
    for (int h = 0; h < headCount; h++)
    {
      MultilayerPerceptron linear = MultilayerPerceptron(ffnNetworkSpecs, initialBias, initialWeightValue, learningRate, &sigmoid, &dSigmoid);
      qkvLinears[i].push_back(linear);
    }
  }

  this->qkvLinears = qkvLinears;
}

vector<double> softmax(vector<double> input)
{
  vector<double> result(input.size(), 0);

  double expSum = expSumVector(input);
  for (int i; i < input.size(); i++)
  {
    std::exp(input[i]) / expSum;
  }

  return result;
}

vector<vector<double>> softmax(vector<vector<double>> input)
{
  vector<vector<double>> result;

  for (int i; i < input.size(); i++)
  {
    result.push_back(softmax(input[i]));
  }

  return result;
}

vector<vector<double>> Encoder::scaledDotProductAttention(vector<vector<double>> K, vector<vector<double>> Q, vector<vector<double>> V, int dim)
{
  vector<vector<double>> QK = matMul(Q, K);
  vector<vector<double>> QK_t = transpose(QK);
  vector<vector<double>> scaled_QK_t = scalarMultiplyMatrix((1 / dim), QK_t);

  return matMul(softmax(scaled_QK_t), V);
}

vector<double> Encoder::multiHeadAttention(vector<double> K, vector<double> Q, vector<double> V)
{
  vector<vector<double>> concatenated(0);
  // Project through linear
  for (int h = 0; h < this->heads; h++)
  {
    vector<vector<double>> QW = this->qkvLinears[h][0].forward(Q);
    vector<vector<double>> KW = this->qkvLinears[h][1].forward(K);
    vector<vector<double>> VW = this->qkvLinears[h][2].forward(V);

    // Give projections to sdpa
    vector<vector<double>> sdpaOut = scaledDotProductAttention(KW, QW, VW, this->d_model);

    // concatenate sdpa outputs
    concatenated.insert(concatenated.end(), sdpaOut.begin(), sdpaOut.end());
  }

  // return Projection of concatenation through linear layer

  return this->ffn.forward(concatenated);
}

vector<double> Encoder::addAndNorm(vector<double> v1, vector<double> v2)
{
  return vectorNormalization(vectorAddition(v1, v2));
}

// TODO: Implement Blocks
vector<double> Encoder::forward(vector<double> x)
{
  vector<double> skip = x;
  vector<double> mmhaOut = multiHeadAttention(x, x, x);
  vector<double> addNorm = addAndNorm(mmhaOut, skip);

  skip = addNorm;
  vector<double> ffnOut = this->mmhaFfn.forward(addNorm);
  addNorm = addAndNorm(ffnOut, skip);

  return addNorm;
}

void Encoder::train(vector<double> x, vector<double> y)
{
  return;
}

double Encoder::lossFunction(vector<double> y, vector<double> y_pred)
{
  return 1.;
}