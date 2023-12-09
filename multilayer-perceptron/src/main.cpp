using namespace std;
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <math.h> 
#include <string>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cfloat>
#include "utils.h"


int main(int argc, char *argv[])
{
    // We want the perceptron to give and-gate logic:
    // input :  output
    // 1 1      1
    // 1 0      0
    // 0 1      0
    // 0 0      0
    Neuron perceptron_1 = Neuron(1.0, .5, std::vector<double>{-1, 1});
    Neuron perceptron_2 = Neuron(1.0, .5, std::vector<double>{-1, 1});
    Neuron perceptron_3 = Neuron(1.0, .5, std::vector<double>{-1, 1});
    Neuron perceptron_4 = Neuron(1.0, .5, std::vector<double>{-1, 1});

    std::vector<std::vector<double>> X = std::vector<std::vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    std::vector<std::vector<double>> y = std::vector<std::vector<double>>{{1},      {1},    {1},    {0}};

    // Training
    for(int i=0;i<10000;i++) {
        for(int i=0;i<X.size();i++) perceptron.train(X[i], y[i]);
    }

    for (int i = 0; i<X.size(); i++) {
        double pred = perceptron.forward_propagate(X[i]);
        std::cout << "pred: " << round(pred) << std::endl;
    }

    printVector(perceptron.weights, "Weights after training");
    std::cout << perceptron.bias << std::endl;

    return 0;
}

