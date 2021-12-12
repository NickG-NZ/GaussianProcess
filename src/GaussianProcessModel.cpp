/**
 * Author: Nick Goodson
 * Date: 12/2021
 */
#include "GaussianProcess/GaussianProcessModel.hpp"
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;


GPModelDataSet::GPModelDataSet() :
    Nx_(0),
    size_(0),
    noiseVariance_(0)
{
}

GPModelDataSet::GPModelDataSet(const MatrixXd& xData, const VectorXd& yData, double noiseVariance) :
    xData_(xData),
    yData_(yData),
    noiseVariance_(noiseVariance)
{
    if (xData_.rows() != yData_.size()) {
        throw std::runtime_error(std::string(__func__) + ": Dataset dimensions are inconsistent");
    }

    if (noiseVariance < 0) {
        throw std::runtime_error(std::string(__func__) + ": Noise variance must be non-negative");
    }

    Nx_ = xData.cols();
    size_ = xData_.rows();

}

GPModelDataSet::GPModelDataSet(std::string_view datafile)
{
    // TODO: Implement using fstream
    throw std::runtime_error(std::string(__func__) + ": Not Implemented");
}

void GPModelDataSet::append(const GPModelDataSet& data)
{
    if (data.Nx_ != Nx_) { 
        throw std::runtime_error(std::string(__func__) + ": Cannot append data, X shape doesn't match");
    }

    size_ += data.size_;
    xData_.conservativeResize(size_, Nx_);
    yData_.conservativeResize(size_);

    xData_.bottomRows(data.size_) = data.xData_;
    yData_.tail(data.size_) = data.yData_;
}

void GPModelDataSet::append(std::string_view datafile)
{
    // TODO: implement this
    throw std::runtime_error(std::string(__func__) + ": Not Implemented");
}


GaussianProcessModel::GaussianProcessModel(KernelFunction trainingKernelFunc, KernelFunction inferenceKernelFunc) :
    trainingKernelFunc_(std::move(trainingKernelFunc)),
    inferenceKernelFunc_(std::move(inferenceKernelFunc))
{
}

void GaussianProcessModel::fit(const GPModelDataSet& data)
{
    // Save data for inference
    if (data.size() < 1) {
        throw std::runtime_error(std::string(__func__) + ": Cannot fit to empty dataset");
    }

    trainingDataSet_ = data;

    // TODO: Implement hyperparameter optimization
    // Use auto differentiation lib or finite diffs
    // Have to recompute kernel with different parameters, then compute log likelihood
    computeTrainingKernel_();

    fitted_ = true;
}

void GaussianProcessModel::updateFit(const GPModelDataSet& data)
{   
    // If no prior data has been added, call the basic fit
    if (!fitted_) {
        fit(data);
        return;
    }
    
    if (data.Nx() != trainingDataSet_.Nx()) {
        throw std::runtime_error(std::string(__func__) + ": Additional data must have same input dimension as existing model");
    }

    // update the kernel
    updateTrainingKernel_(data);

    // update training dataset
    trainingDataSet_.append(data);
}

double GaussianProcessModel::predict(const VectorXd& xInput)
{
    if (!fitted_) {
        throw std::runtime_error(std::string(__func__) + ": No model fitted");
    }

    if (xInput.size() != trainingDataSet_.Nx()) {
       throw std::runtime_error(std::string(__func__) + ": Invalid input size for prediction");
    }
   
   // compute kernel between prediction input and training data (k*)
    inputAndDataKernel_.resize(modelSize());  
    for (Index i = 0; i < modelSize(); ++i) {
        inputAndDataKernel_(i) = inferenceKernelFunc_(trainingDataSet_.xData(i), xInput);
    }

    // compute prediction input kernel with itself (k**)
    inputKernel_ = inferenceKernelFunc_(xInput, xInput);

    // mean (should be a scalar)
    return (inputAndDataKernel_.transpose() * kernelLLT_.solve(trainingDataSet_.yData())).value();
}

double GaussianProcessModel::predict(const VectorXd& xInput, double& predictionVariance)
{
    double mean = predict(xInput);
    predictionVariance = inputKernel_ - (inputAndDataKernel_.transpose() * kernelLLT_.solve(inputAndDataKernel_)).value();  

    return mean;
}

void GaussianProcessModel::saveModel(std::string_view filePath)
{
    // TODO: Implement this
    // Save training data, kernel and hyperparameters to a .gpfit file
    throw std::runtime_error(std::string(__func__) + ": Not implemented");
}

void GaussianProcessModel::loadModel(std::string_view filePath)
{
    // TODO: Implement this
    throw std::runtime_error(std::string(__func__) + ": Not implemented");
}

void GaussianProcessModel::computeTrainingKernel_(void)
{
    kernel_.resize(modelSize(), modelSize());

    for (Index i = 0; i < modelSize(); ++i) {
        for (Index j = i; j < modelSize(); ++j) {
            kernel_(i, j) = trainingKernelFunc_(trainingDataSet_.xData(i), trainingDataSet_.xData(j));
        }
    }
    kernel_.triangularView<Eigen::StrictlyLower>() = kernel_.transpose();
    
    // Add the data noise
    kernel_ += trainingDataSet_.noiseVariance() * MatrixXd::Identity(kernel_.rows(), kernel_.cols());

    // factorize the kernel
    kernelLLT_.compute(kernel_);
}

void GaussianProcessModel::updateTrainingKernel_(const GPModelDataSet& data)
{
    int nUpdated = modelSize() + data.size();
    kernel_.conservativeResize(nUpdated, nUpdated);

    // New data requires two new blocks to be computed in the kernel, B and C
    // newKernel = |A  B|  where A is the existing kernel
    //             |B' C| 

    // new diagonal block, C
    MatrixXd Cblock(data.size(), data.size());
    for (Index i = 0; i < data.size(); ++i) {
        for (Index j = i; j < data.size(); ++j) {
            Cblock(i, j) = trainingKernelFunc_(data.xData(i), data.xData(j));
        }
    }
    Cblock.triangularView<Eigen::StrictlyLower>() = Cblock.transpose();
    kernel_.bottomRightCorner(data.size(), data.size()) = Cblock;

    // new off diagonal block, B
    MatrixXd Bblock(modelSize(), data.size());
    for (Index i = 0; i < modelSize(); ++i) {
        for (Index j = 0; j < data.size(); ++j) {
            Bblock(i, j) = trainingKernelFunc_(trainingDataSet_.xData(i), data.xData(j));
        }
    }
    kernel_.topRightCorner(modelSize(), data.size()) = Bblock;
    kernel_.bottomLeftCorner(data.size(), modelSize()) = Bblock.transpose();

    // Add the data noise
    kernel_ += trainingDataSet_.noiseVariance() * MatrixXd::Identity(nUpdated, nUpdated);

    // factorize the kernel
    kernelLLT_.compute(kernel_);
}
