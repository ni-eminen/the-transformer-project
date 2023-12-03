#include <iostream>
#include <vector>
#include <string>

void printVector(std::vector<double> v, std::string title = "vector") {
    std::cout << std::endl << title << ": [";
    for(double x : v) {
        std::cout << x << ", ";
    }
    std::cout << "]" << std::endl;
}
