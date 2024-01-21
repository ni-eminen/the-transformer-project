using namespace std;
#include <string>
#include "Types.hpp"

double scaledDotProductAttention(vector<double> K, vector<double> Q, vector<double> V);

double multiHeadAttention(vector<double> K, vector<double> Q, vector<double> V);

double maskedMultiHeadAttention(vector<double> K, vector<double> Q, vector<double> V);

vector<double> addAndNorm(vector<vector<double> > V);

vector<double> softmax(vector<double> v);

vector<double> positionEncoding(vector<double> v);

void print(std::string x);

vector<double> forward(vector<double> inputs);

void train(vector<double> x, vector<double> y);

double lossFunction(vector<double> y, vector<double> y_pred);