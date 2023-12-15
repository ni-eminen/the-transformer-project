
#include <cmath>
#include <iostream>
#include <vector>
#include "../../utils/utils.h"
#include "perceptron.cpp"


int main(int argc, char *argv[]) {
    // We want the perceptron to give and-gate logic:
    // input :  output
    // 1 1      1
    // 1 0      0
    // 0 1      0
    // 0 0      0
    Perceptron perceptron = Perceptron(1.0, .5, std::vector<double>{-1, 1});

    std::vector<std::vector<double>> X = std::vector<std::vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    std::vector<std::vector<double>> y = std::vector<std::vector<double>>{{1},      {1},    {1},    {0}};

    // Training
    for(int i=0;i<10000;i++) {
        for(int i=0;i<X.size();i++) perceptron.train(X[i], y[i]);
    }

    for (int i = 0; i<X.size(); i++) {
        double pred = perceptron.forward(X[i]);
        std::cout << "pred: " << round(pred) << std::endl;
    }

    printVector(perceptron.weights, "Weights after training");
    std::cout << perceptron.bias << std::endl;

    return 0;
}

