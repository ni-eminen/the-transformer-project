
#include <vector>
#include "../../perceptron/src/perceptron.cpp"


int main(int argc, char *argv[])
{
    layer_d = 10;
    layer_amt = 4;
    initial_bias = 1;
    
    std::vector<std::vector<double>> X = std::vector<std::vector<double>>{{1,1},    {1,0},  {0,1},  {0,0}};
    std::vector<std::vector<double>> y = std::vector<std::vector<double>>{{1},      {1},    {1},    {0}};

    MultilayerPerceptron mlp = MultilayerPerceptron(layer_d, layer_amt, initial_bias);
    
    for (int i; i<X.size(); i++) {
        mlp.train(X, y);
    }  

    double prediction = mlp.forward(std::vector<double>{1, 0, 0, 0})
    cout << "prediction" << prediction << endl;





    return 0;
}
