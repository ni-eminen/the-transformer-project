
#include <vector>
#include <iomanip>
#include "MultilayerPerceptron.hpp"
#include "utils.hpp"
#include "Types.hpp"

int main(int argc, char *argv[])
{
    double initialBias = 1;
    double initialWeightValue = 1;
    double learningRate = .0001;
    vector<int> networkSpecs = vector<int>{3, 3, 3, 1};

    vector<vector<double> > X = vector<vector<double> >{
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {1, 0, 0},
        {1, 1, 0},
        {1, 0, 1},
        {1, 1, 0},
        {0, 1, 0},
        {0, 1, 1},
    };
    vector<vector<double> > y = vector<vector<double> >{
        {0},
        {1},
        {0},
        {0},
        {1},
        {0},
        {1},
        {1},
        {1},
    };

    MultilayerPerceptron mlp = MultilayerPerceptron(networkSpecs, initialBias, initialWeightValue, learningRate);

    // Training
    for (int i = 0; i < 1000; i++)
    {
        for (int i = 0; i < 100; i++)
        {
            for (int j = 0; j < X.size(); j++)
            {
                mlp.train(X[j], y[j]);
            }
        }
        std::cout << "-----------" << std::endl;
        printMatrix(mlp.weights[1], "first hidden layer: ");
        printMatrix(mlp.weights[1], "");
        printMatrix(mlp.weights[1], "second hidden layer: ");
        printMatrix(mlp.weights[2], "");
        std::cout << std::endl;
        std::cout << "predictions: " << std::endl;
        for (int i = 0; i < X.size(); i++)
        {
            std::cout << std::fixed;
            std::cout << std::setprecision(2);
            double pred = mlp.forward(X[i])[0];
            std::string correct = "incorrect";
            if (round(pred) == y[i][0])
            {
                correct = "Correct";
            }
            std::cout << "input: [ " << (int)round(X[i][0]) << ", " << (int)round(X[i][1]) << ", " << (int)round(X[i][2]) << " ], y_pred: " << pred << " "
                      << "y: " << y[i][0] << ": " << correct << std::endl;
        }
    }

    return 0;
}
