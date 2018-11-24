#ifndef HAMMING_NEURAL_NETWORK_HPP
#define HAMMING_NEURAL_NETWORK_HPP

#include <cstdlib>
#include <vector>
#include <numeric>

class HammingNeuralNetwork
{
public:
    HammingNeuralNetwork();

    void addSampleToLearningDataSet(
            const std::vector< double >& input,
            const std::vector< double >& target );
    void adjustConnectionsWeights();
    std::vector< double > recognizeSample( const std::vector< double >& input );

    size_t getImageLinearSize() const;
    void setLinearSize( const size_t imageLinearSize );

    void clear();

private:
    static const double epsilon;

    size_t imageLinearSize; // image_size.width * image_size.height;

    double randomShittyParameter; // image_linear_size / 2.0;

    std::vector< std::pair< std::vector< double >, std::vector< double > > > samplesMatrix;
    std::vector< std::vector< double > > weightsMatrix;
    std::vector< std::vector< double > > feedbackMatrix;

    void updateWeightsMatrix();
    void updateFeedbackMatrix();

    inline double activation(double arg) const;

    inline double norm(const std::vector< double>& vector) const;
};

#endif // HAMMING_NEURAL_NETWORK_HPP
