
#include <iomanip>
#include "MultilayerPerceptron.hpp"
#include "utils.hpp"
#include "Types.hpp"

int main(int argc, char *argv[])
{
    double initialBias = 1;
    double initialWeightValue = 1;
    double learningRate = 1.5E-4;
    vector<int> networkSpecs = vector<int>{2, 10, 8, 1};

    vector<vector<double> > X = vector<vector<double> >{
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1},
    };
    vector<vector<double> > y = vector<vector<double> >{
        {0},
        {0},
        {0},
        {1},
    };

    MultilayerPerceptron mlp = MultilayerPerceptron(networkSpecs, initialBias, initialWeightValue, learningRate, &sigmoid, &dSigmoid);

    // Training
    for (int i = 0; i < 1000; i++)
    {
        for (int i = 0; i < 500; i++)
        {
            for (int j = 0; j < X.size(); j++)
            {
                mlp.train(X[j], y[j]);
            }
        }
        std::cout << "-----------" << std::endl;
        printMatrix(mlp.weights[1], "first hidden layer: ");
        printMatrix(mlp.weights[2], "output layer: ");
        std::cout << std::endl;
        std::cout << "Epoch: " << i << std::endl;
        std::cout << "predictions: " << std::endl;
        for (int z = 0; z < X.size(); z++)
        {
            std::cout << std::fixed;
            std::cout << std::setprecision(2);
            double pred = mlp.forward(X[z])[0];
            std::string correct = "incorrect";
            if (round(pred) == y[z][0])
            {
                correct = "Correct";
            }
            std::cout << "input: [ " << (int)round(X[z][0]) << ", " << (int)round(X[z][1]) << " ], y_pred: " << pred << " "
                      << "y: " << y[z][0] << ": " << correct << std::endl;
        }
    }

    return 0;
}
