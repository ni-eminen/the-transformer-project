#include <ostream>
#include <iostream>
#include <string>
#include <vector>
#include "LinearAlgebra.hpp"
#include "utils.hpp"
using std::vector;
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
        result.push_back(resultPerRow);
    }

    return result;
}


vector<double> matMul(vector<double> a, matrix B)
{
    matrix A = matrix(1, a);

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
        result.push_back(resultPerRow);
    }

    return result[0];
}


vector<double> matMul(std::vector<double> a, std::vector<double> b)
{
    matrix A = matrix(1, a);
    matrix B = matrix(1, b);

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
        result.push_back(resultPerRow);
    }

    return result[0];
}


vector<double> matMul(matrix A, std::vector<double> b)
{
    matrix B = matrix(1, b);

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
        result.push_back(resultPerRow);
    }

    return result[0];
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

matrix transpose(matrix A) {
    int rows = A.size();
    int cols = A[0].size();
    matrix result;


    for (int i = 0; i<cols; i++) {
        std::vector<double> newRow;
        for (int j = 0; j<rows; j++) {
            newRow.push_back(A[j][i]);
        }
        result.push_back(newRow);
    }

    return result;
}

vector<double> elementWiseSum(vector<double> a, vector<double> b) {
    vector<double> result;
    for (int i = 0; i<a.size(); i++) {
        result.push_back(a[i] + b[i]);
    }
    return result;
}