
#include <vector>
#include "MultilayerPerceptron.hpp"
#include "utils.hpp"
#include "Types.hpp"

int main(int argc, char *argv[])
{
    int hiddenLayerDim = 24;
    int inputLayerDim = 2;
    int outputLayerDim = 1;
    double initialBias = 1;
    double initialWeightValue = 1;
    double learningRate = .02;

    vector<vector<double> > X = vector<vector<double> >{{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    vector<vector<double> > y = vector<vector<double> >{{1}, {1}, {0}, {1}};

    MultilayerPerceptron mlp = MultilayerPerceptron(initialBias, initialWeightValue, hiddenLayerDim, inputLayerDim, outputLayerDim, learningRate);
    // Training

    for (int i = 0; i < 10; i++)
    {
        for (int i = 0; i < 10000; i++)
        {
            for (int j = 0; j < X.size(); j++)
            {
                mlp.train(X[j], y[j]);
            }
        }
        std::cout << "-----------" << std::endl;
        for (int i = 0; i < X.size(); i++)
        {
            std::cout << round(mlp.forward(X[i])[0]) << std::endl;
        }
    }

    // Results

    return 0;
}
