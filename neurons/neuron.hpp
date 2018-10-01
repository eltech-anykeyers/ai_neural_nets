#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <functional>

#include "abstract_neuron.hpp"

class Neuron : public AbstractNeuron
{
public:
    Neuron( const std::vector< AbstractNeuron* >& inputs,
            const std::function< double(double) >& func );

    virtual double getResult() override;
    virtual double getTarget() override;

private:
    std::vector< std::pair< AbstractNeuron*, double > > inputs;
    std::vector< std::pair< AbstractNeuron*, double > > outputs;
    std::function< double(double) > activation_func;
};

#endif /// NEURON_HPP
