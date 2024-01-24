using namespace std;
#include <string>
#include "Types.hpp"
#include "utils.hpp"
#include "LinearAlgebra.hpp"


vector<double> scaledDotProductAttention(vector<double> K, vector<double> Q, vector<double> V, double dim) {
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