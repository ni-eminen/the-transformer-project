
#include <vector>
#include "MultilayerPerceptron.h"


int main(int argc, char *argv[])
{
    int layerDim = 10;
    int layerAmt = 4;
    double initial_bias = 1;
    double initialWeightValue = 1;
    double learningRate = .5;
    
    
    std::vector<std::vector<double>> X = std::vector<std::vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    std::vector<std::vector<double>> y = std::vector<std::vector<double>>{{1},      {1},    {1},    {0}};

    MultilayerPerceptron mlp = MultilayerPerceptron(initial_bias, initialWeightValue, layerAmt, layerDim, learningRate);
    
    for (int i; i<X.size(); i++) {
        mlp.train(X, y);
    }  

    double prediction = mlp.forward(std::vector<double>{1, 0, 0, 0});
    std::cout << "prediction" << prediction << std::endl;

    return 0;
}
