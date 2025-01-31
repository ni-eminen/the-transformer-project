using namespace std;
#include <string>
#include <cmath>
#include <bits/stdc++.h>
#include "Decoder.hpp"
#include "Types.hpp"
#include "utils.hpp"
#include "LinearAlgebra.hpp"
#include "MultilayerPerceptron.hpp"

Decoder::Decoder(
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
  // TODO: Give activation function to Decoder as argument
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

void Decoder::mask(vector<vector<double>> &A)
{
  int i = 0;
  for (vector<double> &v : A)
  {
    if (v.size() > i + 1)
    {
      fill(v.begin() + i + 1, v.end(), 0);
    }
  }
}

vector<vector<double>> Decoder::scaledDotProductAttention(
    const vector<vector<double>> &K,
    const vector<vector<double>> &Q,
    const vector<vector<double>> &V,
    int dim,
    bool mask)
{
  vector<vector<double>> QK = matMul(Q, transpose(K));
  double sqrt_dk = std::sqrt(dim);
  vector<vector<double>> scaled = scalarMultiplyMatrix(1 / sqrt_dk, QK);

  if (mask)
  {
  }

  vector<vector<double>> smax = softmax(scaled);

  return matMul(smax, V);
}

vector<double> Decoder::multiHeadAttention(const vector<double> &K, const vector<double> &Q, const vector<double> &V, bool mask)
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

vector<double> Decoder::addAndNorm(vector<double> v1, vector<double> v2)
{
  return vectorNormalization(vectorAddition(v1, v2));
}

// TODO: Implement Blocks
vector<double> Decoder::forward(vector<double> x)
{
  vector<double> skip = x;
  vector<double> mmhaOut = multiHeadAttention(x, x, x);
  vector<double> addNorm = addAndNorm(mmhaOut, skip);

  skip = addNorm;
  vector<double> ffnOut = this->mmhaFfn.forward(addNorm);
  addNorm = addAndNorm(ffnOut, skip);

  return addNorm;
}

void Decoder::train(vector<double> x, vector<double> y)
{
  return;
}

double Decoder::lossFunction(vector<double> y, vector<double> y_pred)
{
  return 1.;
}

int main()
{
  vector<vector<double>> A(10, vector<double>(10, 1));

  printMatrix(A);

  return 0;
}