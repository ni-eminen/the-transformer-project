#include <ostream>
#include <iostream>
#include <string>
#include <vector>
#include "LinearAlgebra.hpp"
#include "utils.hpp"
#include "Types.hpp"

// TODO: Jagged vector<vector<double> > exception handling etc.
vector<vector<double> > matMul(vector<vector<double> > A, vector<vector<double> > B)
{
    if (A[0].size() != B.size())
    {
        printMatrix(A, "A");
        printMatrix(B, "B");
        throw std::invalid_argument("Columns of A and rows of B must be equal.");
    }

    // resulting vector<vector<double> > is A_rows * B_cols
    vector<vector<double> > result;

    // For each row in A
    for (int A_row = 0; A_row < A.size(); A_row++)
    {
        // Generate a weighted sum with each column of B
        std::vector<double> resultPerRow;
        for (int A_col = 0; A_col < B[0].size(); A_col++)
        {
            double weightedSum = 0;
            for (int B_row = 0; B_row < A[0].size(); B_row++)
            {
                weightedSum += A[A_row][B_row] * B[B_row][A_col];
            }
            resultPerRow.push_back(weightedSum);
        }
        result.push_back(resultPerRow);
    }

    return result;
}

vector<vector<double> > matMul(vector<double> a, vector<vector<double> > B)
{
    vector<vector<double> > A = vector<vector<double> >(1, a);
    return matMul(A, B);
}

double matMul(std::vector<double> a, std::vector<double> b)
{
    return weightedSum(a, b);
}

vector<vector<double> > matMul(vector<vector<double> > A, std::vector<double> b)
{
    vector<vector<double> > B = vector<vector<double> >(1, b);
    return matMul(A, B);
}

void printVector(vector<vector<double> > m, std::string label)
{
    std::cout << label << ":" << std::endl;
    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[0].size(); j++)
        {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

vector<vector<double> > axbVector(int a, int b)
{
    vector<vector<double> > result;
    for (int i = 0; i < a; i++)
    {
        result.push_back(vector<double>(b, 0));
    }
    return result;
}

vector<vector<double> > transpose(vector<vector<double> > A)
{
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<double> > result;

    for (int i = 0; i < cols; i++)
    {
        std::vector<double> newRow;
        for (int j = 0; j < rows; j++)
        {
            newRow.push_back(A[j][i]);
        }
        result.push_back(newRow);
    }

    return result;
}

vector<double> elementWiseSum(vector<double> a, vector<double> b)
{
    vector<double> result;
    for (int i = 0; i < a.size(); i++)
    {
        result.push_back(a[i] + b[i]);
    }

    return result;
}