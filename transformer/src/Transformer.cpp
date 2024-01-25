using namespace std;
#include <string>
#include <bits/stdc++.h>
#include "Types.hpp"
#include "utils.hpp"
#include "LinearAlgebra.hpp"

vector<double> softmax(vector<double> input) {
  vector<double> result(input.size(), 0);

  double expSum = expSumVector(input);
  for (int i; i<input.size(); i++) {
    std::exp(input[i]) / expSum;
  }

  return result;
}

vector<vector<double>> softmax(vector<vector<double>> input) {
  vector<vector<double>> result;

  for (int i; i<input.size(); i++) {
    result.push_back(softmax(input[i]));
  }

  return result;
}


vector<vector<double>> scaledDotProductAttention(vector<vector<double>> K, vector<vector<double>> Q, vector<vector<double>> V, int dim) {
  vector<vector<double>> QK = matMul(Q, K);
  vector<vector<double>> QK_t = transpose(QK);
  vector<vector<double>> scaled_QK_t = scalarMultiplyMatrix((1/dim), QK_t);
  return matMul(softmax(scaled_QK_t), V);
}

double multiHeadAttention(vector<double> K, vector<double> Q, vector<double> V) {
  return 0.0;
}

double maskedMultiHeadAttention(vector<double> K, vector<double> Q, vector<double> V) {
  return 0.0;
}

vector<double> addAndNorm(vector<vector<double> > V) {
  return vector<double>();
}

vector<double> softmax(vector<double> v) {
  return vector<double>();
}

vector<double> positionalEncoding(vector<double> v) {
  return vector<double>();
}

vector<double> forward(vector<double> inputs) {
  return vector<double>();
}

void train(vector<double> x, vector<double> y) {
  return ;
}

double lossFunction(vector<double> y, vector<double> y_pred) {
  return ;
}