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
    HebbianNeuralNetwork( size_t width, size_t height,
                          double** connections );
    ~HebbianNeuralNetwork();
    void addLearningDataSet( const std::vector< double >& dataSet,
                             const std::vector< double >& target );
    void learn();
    std::vector< double > test(
        const std::vector< double >& dataSet );
    void clear();
    double** getWeights() const;

protected:
    double compute( size_t neuronIndex, double* input );
    bool compare( double* input, double* target );
    void adjust( double* input, double* target );

private:
    const size_t inputSize;
    const size_t nNeurons;
    double** connections;
    std::vector< std::pair< double*, double* > > data;
    std::function< double( double ) > activation_func;
};

#endif /// HEBBIAN_NEURAL_NETWORK_HPP
