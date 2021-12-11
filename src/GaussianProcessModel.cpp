/**
 * Author: Nick Goodson
 * Date: 12/2021
 */
#include "GaussianProcess/GaussianProcessModel.hpp"
#include <fstream>


using Eigen::MatrixXd;
using Eigen::VectorXd;


GPModelDataSet::GPModelDataSet(const MatrixXd& xData, const VectorXd& yData, double noiseVariance) :
    xData_(xData),
    yData_(yData),
    noiseVariance_(noiseVariance)
{
    if (xData_.cols() != yData_.size()) {
        throw std::runtime_error("Dataset dimensions are inconsistent");
    }

    if (noiseVariance < 0) {
        throw std::runtime_error("Noise variance must be non-negative");
    }

    Nx_ = xData.rows();
    size_ = xData_.cols();

}

GPModelDataSet::GPModelDataSet(std::string_view datafile)
{
    // TODO: Implement using fstream
    throw std::runtime_error("Not Implemented");

}

void GPModelDataSet::append(const GPModelDataSet& data)
{
    if (data.Nx_ != Nx_) {
        throw std::runtime_error("Cannot append data, X shape doesn't match");
    }

    size_ += data.size_;
    xData_.conservativeResize(size_, Nx_);
    yData_.conservativeResize(size_);

    xData_.bottomRows(size_) = data.xData_;
    yData_.tail(size_) = data.yData_;
}

void GPModelDataSet::append(std::string_view datafile)
{
    // TODO: implement this
    throw std::runtime_error("Not Implemented");
}


GaussianProcessModel::GaussianProcessModel(KernelFunction trainingKernelFunc, KernelFunction inferenceKernelFunc) :
    trainingKernelFunc_(std::move(trainingKernelFunc)),
    inferenceKernelFunc_(std::move(inferenceKernelFunc))
{
}

void GaussianProcessModel::fit(const GPModelDataSet& data)
{
    // Save data for inference
    trainingDataSet_ = data;

    // TODO: Implement hyperparameter optimization
    // Use auto differentiation lib or finite diffs
    // Have to recompute kernel with different parameters, then compute log likelihood
    computeTrainingKernel_();

    fitted_ = true;
}

void GaussianProcessModel::updateFit(const GPModelDataSet& data)
{   
    if (data.Nx() != trainingDataSet_.Nx()) {
        throw std::runtime_error("Additional data must have same input dimension as existing model");
    }

    updateTrainingKernel_(data);

    // update training dataset
    trainingDataSet_.append(data);

}

double GaussianProcessModel::predict(double& predictionVariance, const VectorXd& xInput, bool computeVariance /* false */)
{
    // compute kernel between prediction input and training data (k*)
    VectorXd inputAndDataKernel(trainingDataSet_.size());  
    for (Index i = 0; i < trainingDataSet_.size(); ++i) {
        inputAndDataKernel(i) = inferenceKernelFunc_(trainingDataSet_.xData(i), xInput);
    }

    // compute prediction input kernel with itself (k**)
    double inputKernel = inferenceKernelFunc_(xInput, xInput);

    // mean (should be a scalar)
    double mean = (inputAndDataKernel.transpose() * kernelLLT_.solve(trainingDataSet_.yData())).value();

    // variance
    if (computeVariance) {
        predictionVariance = inputKernel - (inputAndDataKernel.transpose() * kernelLLT_.solve(inputAndDataKernel)).value();
    }

    return mean;

}

void GaussianProcessModel::saveModel(std::string_view filePath)
{
    // TODO: Implement this
    // Save training data, kernel and hyperparameters to a .gpfit file
    throw std::runtime_error("Not implemented");
}

void GaussianProcessModel::loadModel(std::string_view filePath)
{
    // TODO: Implement this
    throw std::runtime_error("Not implemented");
}

void GaussianProcessModel::computeTrainingKernel_(void)
{
    kernel_.resize(trainingDataSet_.size(), trainingDataSet_.size());

    for (Index i = 0; i < trainingDataSet_.size(); ++i) {
        for (Index j = i; j < trainingDataSet_.size(); ++j) {
            kernel_(i, j) = trainingKernelFunc_(trainingDataSet_.xData(i), trainingDataSet_.xData(j));
        }
    }
    kernel_.triangularView<Eigen::StrictlyLower>() = kernel_.triangularView<Eigen::StrictlyUpper>();
    
    // Add the data noise
    kernel_ += trainingDataSet_.noiseVariance() * MatrixXd::Identity(trainingDataSet_.size(), trainingDataSet_.size());

    // factorize the kernel
    kernelLLT_.compute(kernel_);
}

void GaussianProcessModel::updateTrainingKernel_(const GPModelDataSet& data)
{
    int nUpdated = trainingDataSet_.size() + data.size();
    kernel_.conservativeResize(nUpdated, nUpdated);

    // New data requires two new blocks to be computed in the kernel, B and C
    // newKernel = |A  B|  where A is the existing kernel
    //             |B' C| 

    // new diagonal block, C
    for (Index i = 0; i < data.size(); ++i) {
        for (Index j = i; j < data.size(); ++j) {
            kernel_(i + data.size(), j + data.size()) = trainingKernelFunc_(data.xData(i), data.xData(j));
        }
    }

    // new off diagonal block, B
    Eigen::MatrixXd Bblock(trainingDataSet_.size(), data.size());
    for (Index i = 0; i < trainingDataSet_.size(); ++i) {
        for (Index j = 0; j < data.size(); ++j) {
            Bblock(i, j) = trainingKernelFunc_(trainingDataSet_.xData(i), data.xData(j));
        }
    }
    kernel_.block(0, trainingDataSet_.size(), trainingDataSet_.size(), data.size()) = Bblock;
    kernel_.block(trainingDataSet_.size(), 0, data.size(), trainingDataSet_.size()) = Bblock.transpose();

    // Add the data noise
    kernel_ += trainingDataSet_.noiseVariance() * MatrixXd::Identity(nUpdated, nUpdated);

    // factorize the kernel
    kernelLLT_.compute(kernel_);
}
