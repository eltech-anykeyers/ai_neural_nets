#ifndef HEBBIAN_NEURAL_NETWORK_HPP
#define HEBBIAN_NEURAL_NETWORK_HPP

#include <vector>
#include <functional>
#include <iterator>

class HebbianNeuralNetwork
{
public:
    HebbianNeuralNetwork() = delete;
    HebbianNeuralNetwork( size_t inputSize, size_t nNeurons );
    HebbianNeuralNetwork( size_t weightsMatrixWidth, size_t seightsMatrixHeight,
                          double** weightsMatrix );
    ~HebbianNeuralNetwork();
    void addSampleToLearningDataSet( const std::vector< double >& input,
                                     const std::vector< double >& target );
    void adjustConnectionsWeights();
    std::vector< double > recognizeSample( const std::vector< double >& input );
    void clear();
    double** getWeights() const;

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
