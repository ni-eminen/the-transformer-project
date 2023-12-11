using namespace std;
#include <vector>
#include "../../perceptron/src/perceptron.cpp"

class MultilayerPerceptron {
    public:
        vector<double> weights;
        vector<Perceptron> neurons;

        double forwardPropagate(std::vector<double> inputs) {
            double weightedSum = combinationFunction(inputs, weights);
            double activation_output = activationFunction(weightedSum);

            return activation_output;
        }
};


int main(int argc, char *argv[])
{
    Perceptron perceptron_1 = Perceptron(1.0, .5, std::vector<double>{-1, 1});
    Perceptron perceptron_2 = Perceptron(1.0, .5, std::vector<double>{-1, 1});
    Perceptron perceptron_3 = Perceptron(1.0, .5, std::vector<double>{-1, 1});

    std::vector<std::vector<Perceptron>> neurons = {
        {perceptron_1, perceptron_2, perceptron_3}
    };

    return 0;
}

