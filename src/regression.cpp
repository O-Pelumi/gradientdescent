
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <math.h>

#include "readfile.h"

using namespace Eigen;


VectorXd stochasticGD(const MatrixXd& X, const VectorXd& Y)
{
    const size_t allRows = X.rows();
    VectorXd theta = VectorXd::Zero(X.cols());

    for (size_t i = 0; i < allRows; ++i) {
        VectorXd hTheta = X.row(i) * theta;
        VectorXd diff = hTheta - Y.row(i);

        for (size_t j = 0; j < theta.rows(); ++j) {
            theta.row(j) -= (diff * X.row(i).col(j));
        }
    }

    return theta;
}

template <class OutputIterator>
VectorXd batchGD(const MatrixXd& X, const VectorXd& Y, OutputIterator costs, const size_t maxIter = 1000)
{
    const size_t allRows = X.rows();
    VectorXd theta = VectorXd::Zero(X.cols());

    const double acceptableCost = 0.000001;
    double cost = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < maxIter && cost > acceptableCost; ++i) {
        VectorXd hTheta = X * theta;
        VectorXd diff = hTheta - Y;

        auto diff_T = diff.transpose();

        for (size_t j = 0; j < theta.rows(); ++j) {
            theta.row(j) -= (diff_T * X.col(j)) / allRows;
        }

        costs = cost = (diff_T * diff / (2 * allRows))(0);
    }

    return theta;
}

void plotVector(const std::vector<double>& vec)
{
    FILE* gp = popen("gnuplot", "w");
    fprintf(gp, "reset \n");
    // fprintf(gp, "set terminal qt \n");
    fprintf(gp, "set grid \n");
    fprintf(gp, "set title 'Cost Function' \n");
    fprintf(gp, "set xlabel 'Number of Iterations' \n");
    fprintf(gp, "set ylabel 'Cost J' \n");
    fprintf(gp, "set style data lines \n");
    fprintf(gp, "plot '-' lw 1 \n");

    size_t i = 0;
    for (const auto& val : vec) {
        fprintf(gp, "%zu %f \n", i++, val);
    }
    fprintf(gp, "e\n");

    fflush(gp);

    puts("Hit enter to continue");
    getchar();
    pclose(gp);
}

int main(int argc, char* argv[])
{
    constexpr int rows = 13;
    constexpr int columns = 3;
    MatrixXd data(rows, columns);

    if (argc < 2) {
        printf("usage: %s <data>\n", argv[0]);
        return 0;
    }

    if (readData(argv[1], data) < 0) {
        fprintf(stderr, "Error reading data\n");
        return 0;
    }

    data.conservativeResize(NoChange, data.cols() + 1);
    data.col(columns) = data.col(columns - 1); 
    data.col(columns - 1).setOnes();            // Set intercept to column to 1

    std::vector<double> costs;
    VectorXd theta = batchGD(data.block<rows, columns>(0, 0), data.col(columns), std::back_inserter(costs));
    std::cout << "theta: " << theta.transpose() << std::endl;
    std::cout << "iterations: " << costs.size() << std::endl;

    plotVector(costs);

    // std::cout << "Answer ->\n" << (X.transpose() * X).ldlt().solve(X.transpose() * Y) << std::endl;
    // std::cout << "Answer ->\n" << X.completeOrthogonalDecomposition().solve(Y) << std::endl;

    return 0;
}


