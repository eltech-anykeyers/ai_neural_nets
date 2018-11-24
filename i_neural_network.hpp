#ifndef I_NEURAL_NETWORK_HPP
#define I_NEURAL_NETWORK_HPP

#include <vector>

class INeuralNetwork
{
public:
    struct Matrix
    {
        double** matrix;
        size_t width;
        size_t height;
    };

    INeuralNetwork() = default;
    virtual ~INeuralNetwork() = default;

    virtual void addSampleToLearningDataSet(
            const std::vector< double >& input,
            const std::vector< double >& target ) = 0;
    virtual void adjustConnectionsWeights() = 0;
    virtual std::vector< double > recognizeSample(
            const std::vector< double >& input ) = 0;
    virtual void clear() = 0;
    virtual std::vector< Matrix > getWeightsMatrices() const = 0;
};

#endif /// I_NEURAL_NETWORK_HPP
