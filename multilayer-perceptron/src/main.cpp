
#include <vector>
#include "MultilayerPerceptron.hpp"
#include "../../utils/utils.hpp"
using std::vector;


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

    vector<double> outputs = mlp.forward(std::vector<double>{-100, 0, 10, 4});
    printVector(outputs, "output");

    
    // for (int i; i<X.size(); i++) {
    //     mlp.train(X, y);
    // }

    // double prediction = mlp.forward(std::vector<double>{1, 0, 0, 0});
    // std::cout << "prediction" << prediction << std::endl;

    return 0;
}
