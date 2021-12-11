/**
 * Author: Nick Goodson
 * Date: 12/2021
 */
#include <string_view>
#include <functional>
#include <Eigen/Dense>


class GPModelDataSet
{
public:

    GPModelDataSet() = default;
    GPModelDataSet(const Eigen::MatrixXd& xData, const Eigen::VectorXd& yData, double noiseVariance);
    GPModelDataSet(std::string_view datafile);

    const Eigen::MatrixXd& xData(void) const { return xData_; }
    const Eigen::MatrixXd& xData(Eigen::Index idx) const { return xData_.row(idx); }
    const Eigen::VectorXd& yData(void) const { return yData_; }
    double noiseVariance(void) const { return noiseVariance_; }
    int size(void) const { return size_; }
    int Nx(void) const { return Nx_; }

    void append(const GPModelDataSet& data);
    void append(std::string_view datafile);

protected:
    Eigen::MatrixXd xData_;
    Eigen::VectorXd yData_;
    double noiseVariance_;           // variance for data (assumes IId samples with Guassian noise)

    int Nx_;                // input vector dimension
    int size_;              // number of data points   

};


class GaussianProcessModel
{
public:
    using KernelFunction = std::function<double(const Eigen::VectorXd&, const Eigen::VectorXd&)>;
    using Index = Eigen::Index;

    GaussianProcessModel() = delete;
    GaussianProcessModel(KernelFunction trainingKernelFunc, KernelFunction inferenceKernelFunc);

    void setTrainingKernelFunc(KernelFunction kernelFunc) { trainingKernelFunc_ = kernelFunc; }
    void setInferenceKernelFunc(KernelFunction kernelFunc) { inferenceKernelFunc_ = kernelFunc; }

    /**
     * Perform hyperparameter optimization and save the training data marginal kernel
     */ 
    void fit(const GPModelDataSet& data);
    void updateFit(const GPModelDataSet& data);

    /**
     * Performs prediction for a single test point
     * If computeVariance is true, the conditional variance of the prediction given the data will be
     * assigned to the predictionVariance argument
     */ 
    double predict(double& predictionVariance, const Eigen::VectorXd& xInput, bool computeVariance = false);

    /**
     * Save and load the model
     * Stores the training data, kernel and optimized hyperparameters
     * The kernelFunctions still need to be provided for inference and adding more data
     */ 
    void saveModel(std::string_view filePath);
    void loadModel(std::string_view filePath);
    
private:
    void computeTrainingKernel_(void);
    void updateTrainingKernel_(const GPModelDataSet& data);


    Eigen::MatrixXd kernel_;                // training data kernel
    Eigen::LLT<Eigen::MatrixXd> kernelLLT_; // Cholesky decomposition

    GPModelDataSet trainingDataSet_;

    KernelFunction trainingKernelFunc_;
    KernelFunction inferenceKernelFunc_;

    bool fitted_ = false;

    // TODO: The kernel function should have hyperparameters, theta to optimize
    // TOOD: optimize argmax(theta) log p(y|X;theta) during fitting

};