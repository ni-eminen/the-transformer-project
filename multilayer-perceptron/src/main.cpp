
#include <vector>
#include <iomanip>
#include "MultilayerPerceptron.hpp"
#include "utils.hpp"
#include "Types.hpp"

int main(int argc, char *argv[])
{
    double initialBias = 1;
    double initialWeightValue = 1;
    double learningRate = .01;
    vector<int> networkSpecs = vector<int>{3, 4, 4, 4, 1};

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
        for (int i = 0; i < X.size(); i++)
        {
            std::cout << std::fixed;
            std::cout << std::setprecision(2);
            std::cout << mlp.forward(X[i])[0] << std::endl;
        }
    }

    return 0;
}
