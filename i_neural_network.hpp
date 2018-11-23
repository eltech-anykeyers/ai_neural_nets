#ifndef I_NEURAL_NETWORK_HPP
#define I_NEURAL_NETWORK_HPP

#include <vector>

class INeuralNetwork
{
public:
    INeuralNetwork() = default;
    virtual ~INeuralNetwork() = default;

    virtual void addSampleToLearningDataSet(
            const std::vector< double >& input,
            const std::vector< double >& target ) = 0;
    virtual void adjustConnectionsWeights() = 0;
    virtual std::vector< double > recognizeSample(
            const std::vector< double >& input ) = 0;
    virtual void clear() = 0;
    virtual double** getWeights() const = 0;
};

#endif /// I_NEURAL_NETWORK_HPP
