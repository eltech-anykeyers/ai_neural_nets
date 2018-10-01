#ifndef HEBBIANNEURALNETWORK_HPP
#define HEBBIANNEURALNETWORK_HPP

#include <vector>

#include "neurons/input_neuron.hpp"
#include "neurons/target_neuron.hpp"
#include "neurons/neuron.hpp"

class HebbianNeuralNetwork
{
public:
    using InputLayer = std::vector< InputNeuron >;
    using NeuronsLayer = std::vector< Neuron >;
    using TargetLayer = std::vector< TargetNeuron >;
    using DataSet = std::vector< double >;

    HebbianNeuralNetwork( size_t inputSize, size_t nNeurons );
    void addLearningDataSet( const std::vector< double >& dataSet,
                             const std::vector< double >& target );
    void learn();
    bool test( const std::vector< double >& dataSet,
               const std::vector< double >& target );

protected:
    bool compare();
    void setDataSet( const std::vector< double >& dataSet,
                     const std::vector< double >& target );

private:
    size_t inputSize;
    size_t nNeurons;
    InputLayer inputLayer;
    NeuronsLayer neuronsLayer;
    TargetLayer targetLayer;
    std::vector< std::pair< DataSet, DataSet > > data;
};

#endif /// HEBBIANNEURALNETWORK_HPP
