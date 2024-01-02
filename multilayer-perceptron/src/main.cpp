
#include <vector>
#include "MultilayerPerceptron.hpp"
using std::vector;


int main(int argc, char *argv[])
{
    int hiddenLayerDim = 10;
    int inputLayerDim = 2;
    int outputLayerDim = 1;
    double initialBias = 1;
    double initialWeightValue = 1;
    double learningRate = .5;
    
    
    vector<vector<double>> X = vector<vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    vector<vector<double>> y = vector<vector<double>>{{1},      {1},    {1},    {0}};

    MultilayerPerceptron mlp = MultilayerPerceptron(initialBias, initialWeightValue, hiddenLayerDim, inputLayerDim, outputLayerDim, learningRate);
    
    // for (int i; i<X.size(); i++) {
    //     mlp.train(X, y);
    // }  

    // double prediction = mlp.forward(std::vector<double>{1, 0, 0, 0});
    // std::cout << "prediction" << prediction << std::endl;

    return 0;
}
