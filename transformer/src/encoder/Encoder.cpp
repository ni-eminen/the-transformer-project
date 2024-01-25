using namespace std;
#include <string>
#include <bits/stdc++.h>
#include "Encoder.hpp"
#include "Types.hpp"
#include "utils.hpp"
#include "LinearAlgebra.hpp"
#include "MultilayerPerceptron.cpp"

Encoder::Encoder(vector<int> networkSpecs,
                 double initialBias,
                 double initialWeightValue,
                 double learningRate,
                 int heads,
                 int d_model)
{
  MultilayerPerceptron ffn = MultilayerPerceptron(networkSpecs, initialBias, initialWeightValue, learningRate, &sigmoid, &dSigmoid);

  vector<vector<MultilayerPerceptron>> qkvLinears = vector<vector<MultilayerPerceptron>>(3, vector<MultilayerPerceptron>());
  for (int i = 0; i < 3; i++)
  {
    for (int h = 0; h < heads; h++)
    {
      MultilayerPerceptron linear = MultilayerPerceptron(networkSpecs, initialBias, initialWeightValue, learningRate, &sigmoid, &dSigmoid);
      qkvLinears[i].push_back(linear);
    }
  }
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

vector<vector<double>> scaledDotProductAttention(vector<vector<double>> K, vector<vector<double>> Q, vector<vector<double>> V, int dim)
{
  vector<vector<double>> QK = matMul(Q, K);
  vector<vector<double>> QK_t = transpose(QK);
  vector<vector<double>> scaled_QK_t = scalarMultiplyMatrix((1 / dim), QK_t);
  return matMul(softmax(scaled_QK_t), V);
}

double multiHeadAttention(vector<double> K, vector<double> Q, vector<double> V)
{
  // Project through linear

  // Give projections to sdpa
  // concatenate
  // return Projection of concatenation through linear layer

  return 0.0;
}

double maskedMultiHeadAttention(vector<double> K, vector<double> Q, vector<double> V)
{
  return 0.0;
}

vector<double> addAndNorm(vector<double> v1, vector<double> v2)
{
  return vectorNormalization(vectorAddition(v1, v2));
}

vector<double> forward(vector<double> inputs)
{
  return vector<double>();
}

void train(vector<double> x, vector<double> y)
{
  return;
}

double lossFunction(vector<double> y, vector<double> y_pred)
{
  return 1.;
}