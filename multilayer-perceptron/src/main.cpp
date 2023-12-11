using namespace std;
#include <vector>
#include "../../perceptron/src/perceptron.cpp"

class MultilayerPerceptron {
    public:
        vector<double> weights;
        vector<Perceptron> neurons;
};


int main(int argc, char *argv[])
{
    Perceptron perceptron_1 = Perceptron(1.0, .5, std::vector<double>{-1, 1});
    Perceptron perceptron_2 = Perceptron(1.0, .5, std::vector<double>{-1, 1});
    Perceptron perceptron_3 = Perceptron(1.0, .5, std::vector<double>{-1, 1});

    return 0;
}

