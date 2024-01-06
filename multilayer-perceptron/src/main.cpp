
#include <vector>
#include "MultilayerPerceptron.hpp"
#include "utils.hpp"
#include "Types.hpp"

int main(int argc, char *argv[])
{
    int hiddenLayerDim = 10;
    int inputLayerDim = 2;
    int outputLayerDim = 1;
    double initialBias = 1;
    double initialWeightValue = 1;
    double learningRate = .5;

    vector<vector<double> > X = vector<vector<double> >{{1, 1}, {1, 0}, {0, 1}, {0, 0}};
    vector<vector<double> > y = vector<vector<double> >{{1}, {1}, {1}, {0}};
    MultilayerPerceptron mlp = MultilayerPerceptron(initialBias, initialWeightValue, hiddenLayerDim, inputLayerDim, outputLayerDim, learningRate);
    mlp.train(X[0], y[0]);

    // Training
    // for (int i = 0; i < 10000; i++)
    // {
    //     for (int j = 0; j < X.size(); j++)
    //     {
    //         mlp.train(X[j], y[j]);
    //     }
    // }

    vector<double> outputs = mlp.forward(vector<double>{-100, 0, 10, 4});
    printVector(outputs, "output");

    return 0;
}
