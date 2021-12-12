/**
 * Test exmaples for GP fit
 */
// #include <GaussianProcess/GaussianProcessModel.hpp>
#include <cmath>
#include <iostream>
#include <vector>
#include "../include/GaussianProcess/GaussianProcessModel.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

GPModelDataSet createDataSet(int nSamples, int inputDimension, double noiseVariance = 0);
double kernelFunc(const VectorXd& x1, const VectorXd& x2);


int main(int argc, char** argv)
{
    // 2D Model
    // ===============================
    GPModelDataSet dataSet2DA = createDataSet(5, 2);
    GaussianProcessModel gpModel2D(kernelFunc, kernelFunc);
    gpModel2D.fit(dataSet2DA);

    // Do single inference
    VectorXd xPred {{0.3, -0.22}};
    std::cout << "Prediction: " << gpModel2D.predict(xPred) << std::endl;

    // Generate and add more data
    GPModelDataSet dataSet2DB = createDataSet(3, 2);
    gpModel2D.updateFit(dataSet2DB);

    // 1D Model (multiple inference)
    // ================================
    GPModelDataSet dataSet1D = createDataSet(5, 1);
    GaussianProcessModel gpModel1D(kernelFunc, kernelFunc);
    gpModel1D.fit(dataSet1D);

    int N = 20;
    std::vector<double> inferencePoints(N);
    std::vector<double> outputs(N);
    double xMin = -1;
    double xMax = 1;
    for (int i = 0; i < N + 1; ++i) {

        double xVal = xMin + (2.0 / static_cast<double>(N)) * static_cast<double>(i);  // linear range of points
        double output = gpModel1D.predict(Eigen::Matrix<double, 1, 1>{xVal});

        inferencePoints.push_back(xVal);
        outputs.push_back(output);
    }

    return 0;
}


GPModelDataSet createDataSet(int nSamples, int inputDimension, double noiseVariance /* 0.0 */)
{
    // Generate data from uniform distribution over [-1:1]
    MatrixXd xData = MatrixXd::Random(nSamples, inputDimension);
    VectorXd yData = VectorXd::Random(nSamples);
    double dataNoiseVariance = noiseVariance;  // arbitrary here

    return GPModelDataSet(xData, yData, dataNoiseVariance);
}

/**
 * Squared exponential (Radial basis) function
 */ 
double kernelFunc(const VectorXd& x1, const VectorXd& x2)
{
    // std::cout << "x1: " << x1 << std::endl;
    // std::cout << "x2: " << x2 << std::endl;

    double value = std::exp(-0.5 * std::pow((x1 - x2).norm(), 2));

    return std::exp(-0.5 * std::pow((x1 - x2).norm(), 2));
}

