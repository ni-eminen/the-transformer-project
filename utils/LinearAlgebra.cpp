#include <ostream>
#include <iostream>
#include <string>
#include <vector>
#include "LinearAlgebra.hpp"
#include "utils.hpp"
using matrix = std::vector<std::vector<double>>;

matrix matMul(matrix A, matrix B)
{
    // resulting matrix is A_rows * B_cols
    matrix result;

    // For each row in A
    for (int A_row = 0; A_row<A.size(); A_row++) {
        // Generate a weighted sum with each column of B
        std::vector<double> resultPerRow;
        for (int A_col = 0; A_col<B[0].size(); A_col++) {
            double weightedSum = 0;
            for (int B_row = 0; B_row<A[0].size(); B_row++) {
                weightedSum += A[A_row][B_row] * B[B_row][A_col];
            }
            resultPerRow.push_back(weightedSum);
        }
        printVector(resultPerRow, "result per row");
        result.push_back(resultPerRow);
        printMatrix(result, "Result (matrix)");
    }

    return result;
}

void printMatrix(matrix m, std::string label) {
    std::cout << label << ":" << std::endl;
    for (int i = 0; i<m.size(); i++) {
        for (int j = 0; j<m[0].size(); j++) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
}