#ifndef HEBBIAN_NEURAL_NETWORK_HPP
#define HEBBIAN_NEURAL_NETWORK_HPP

#include <vector>
#include <functional>
#include <iterator>

#include "i_neural_network.hpp"

class HebbianNeuralNetwork : public INeuralNetwork
{
public:
    HebbianNeuralNetwork() = delete;
    explicit HebbianNeuralNetwork( size_t inputSize, size_t nNeurons );
    explicit HebbianNeuralNetwork( size_t weightsMatrixWidth, size_t seightsMatrixHeight,
                                   double** weightsMatrix );
    virtual ~HebbianNeuralNetwork() override;
    virtual void addSampleToLearningDataSet(
            const std::vector< double >& input,
            const std::vector< double >& target ) final;
    virtual void adjustConnectionsWeights() final;
    virtual std::vector< double > recognizeSample(
            const std::vector< double >& input ) final;
    virtual void clear() final;
    virtual std::vector< Matrix > getWeightsMatrices() const final;

protected:
    double compute( size_t neuronIndex, double* input );
    bool compare( double* input, double* target );
    void adjust( double* input, double* target );

private:
    const size_t inputSize;
    const size_t nNeurons;
    double** connectionsWeightsMatrix;
    std::vector< std::pair< double*, double* > > data;
    std::function< double( double ) > activation_func;
};

#endif /// HEBBIAN_NEURAL_NETWORK_HPP
