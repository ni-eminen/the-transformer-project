using namespace std;
#include <iostream>
#include <vector>
#include <cmath>
#include "../../utils/utils.h"
#include "perceptron.cpp"
using namespace std;

int main(int argc, char *argv[])
{
    // We want the perceptron to give and-gate logic:
    // input :  output
    // 1 1      1
    // 1 0      0
    // 0 1      0
    // 0 0      0
    Perceptron perceptron = Perceptron(1.0, .5, vector<double>{-1, 1});

    vector<vector<double>> X = vector<vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    vector<vector<double>> y = vector<vector<double>>{{1},      {1},    {1},    {0}};

    // Training
    for(int i=0;i<10000;i++) {
        for(int i=0;i<X.size();i++) perceptron.train(X[i], y[i]);
    }

    for (int i = 0; i<X.size(); i++) {
        double pred = perceptron.forward(X[i]);
        cout << "pred: " << round(pred) << endl;
    }

    printVector(perceptron.weights, "Weights after training");
    cout << perceptron.bias << endl;

    return 0;
}

