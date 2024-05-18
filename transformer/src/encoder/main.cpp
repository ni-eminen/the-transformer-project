
#include <vector>
#include <iomanip>
#include "Encoder.hpp"
#include "utils.hpp"
#include "Types.hpp"

int main(int argc, char *argv[])
{
    double learningRate = .01;
    int heads = 4;
    double d_model = 10;
    vector<int> ffnNetworkSpecs = vector<int>{10, 10};
    vector<int> mmhaFfnNetworkSpecs = vector<int>{10, 10};
    int d_k = 10;
    int d_v = 10;

    Encoder encoder = Encoder(
        learningRate,
        heads,
        d_model,
        ffnNetworkSpecs,
        mmhaFfnNetworkSpecs,
        heads);

    return 0;
}
