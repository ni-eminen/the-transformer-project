#include <ostream>
#include <iostream>
#include <vector>
#include "LinearAlgebra.hpp"
#include "utils.hpp"

#include <chrono>
using namespace std::chrono;

int main(int argc, char *argv[])
{
    for (int i = -10; i < 10; i++)
    {
        std::cout << ReLU(i) << std::endl;
    }

    return 0;
}