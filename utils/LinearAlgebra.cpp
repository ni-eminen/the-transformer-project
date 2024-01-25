#include <ostream>
#include <iostream>
#include <string>
#include <vector>
#include <bits/stdc++.h>
#include<cmath>
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

double sumVector(vector<double> v) {
    double result = 0;
    for (double x : v) {
        result += x;
    }

    return result;
}

double expSumVector(vector<double> v) {
    double result = 0;
    for (double x : v) {
        result += std::exp(x);
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

vector<vector<double>> scalarMultiplyMatrix(double scalar, vector<vector<double>> A) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<double>> result = vector<vector<double>>(rows, vector<double>(cols, 0));

    for (int row = 0; row<rows; row++) {
        for (int col = 0; col<cols; col++) {
            result[row][col] = scalar * A[row][col];
        }
    }

    return result;
}

vector<double> vectorAddition(vector<double> v1, vector<double> v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vectors must be equal size.");
    }

    vector<double> result;

    for (int i = 0; i<v1.size(); i++) {
        result.push_back(v1[i] + v2[i]);
    }

    return result;    
}

vector<double> vectorNormalization(vector<double> v) {
    vector<double> result;
    double length = vectorLength(v);
    for (int i = 0; i<v.size(); i++) {
        result.push_back(v[i] / length);
    }

    return result;
}

double vectorLength(vector<double> v) {
    double result = 0;
    for (int i = 0; i<v.size(); i++) {
        result += pow(v[i], 2);
    }
    return sqrt(result);
}