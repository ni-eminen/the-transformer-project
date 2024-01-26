
#include <vector>
#include <iomanip>
#include "Encoder.hpp"
#include "utils.hpp"
#include "Types.hpp"

int main(int argc, char *argv[])
{
    Encoder encoder = Encoder(
        learningRate,
        heads,
        d_model,
        ffnNetworkSpecs,
        d_k,
        d_v);

    return 0;
}
