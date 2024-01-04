
#include <vector>
#include "MultilayerPerceptron.hpp"
#include "../../utils/utils.hpp"
#include "../../utils/types.hpp"


int main(int argc, char *argv[])
{
    int hiddenLayerDim = 10;
    int inputLayerDim = 4;
    int outputLayerDim = 4;
    double initialBias = 1;
    double initialWeightValue = 1;
    double learningRate = .5;
    
    vector<vector<double>> X = vector<vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    vector<vector<double>> y = vector<vector<double>>{{1},      {1},    {1},    {0}};

    MultilayerPerceptron mlp = MultilayerPerceptron(initialBias, initialWeightValue, hiddenLayerDim, inputLayerDim, outputLayerDim, learningRate);

    // Training
    for(int i=0;i<10000;i++) {
        for(int i=0;i<X.size();i++) mlp.train(X[i], y[i]);
    }

    vector<double> outputs = mlp.forward(std::vector<double>{-100, 0, 10, 4});
    printVector(outputs, "output");

    return 0;
}
