#ifndef HAMMING_NEURAL_NETWORK_HPP
#define HAMMING_NEURAL_NETWORK_HPP

#include <cstdlib>
#include <vector>
#include <numeric>

#include "i_neural_network.hpp"

class HammingNeuralNetwork : public INeuralNetwork
{
public:
    HammingNeuralNetwork() = delete;
    explicit HammingNeuralNetwork( const size_t inputSize, const size_t nNeurons );

    virtual void addSampleToLearningDataSet(
            const std::vector< double >& input,
            const std::vector< double >& target ) final;
    virtual void adjustConnectionsWeights() final;
    virtual std::vector< double > recognizeSample(
            const std::vector< double >& input ) final;
    virtual void clear() final;
    virtual std::vector< Matrix > getWeightsMatrices() const final;

    size_t getImageLinearSize() const;

private:
    static const double epsilon;

    const size_t nNeurons;
    const size_t inputSize; // image_size.width * image_size.height;

    double randomShittyParameter; // image_linear_size / 2.0;

    std::vector< std::pair< std::vector< double >, std::vector< double > > > samplesMatrix;
    std::vector< std::vector< double > > weightsMatrix;
    std::vector< std::vector< double > > feedbackMatrix;

    void updateWeightsMatrix();
    void updateFeedbackMatrix();

    inline double activation( double arg ) const;

    inline double norm( const std::vector< double>& vector ) const;
};

#endif // HAMMING_NEURAL_NETWORK_HPP
