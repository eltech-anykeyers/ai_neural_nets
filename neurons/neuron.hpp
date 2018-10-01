#ifndef NEURON_HPP
#define NEURON_HPP

#include <vector>
#include <functional>

#include "abstract_neuron.hpp"

class Neuron : public AbstractNeuron
{
public:
    Neuron( const std::vector< AbstractNeuron* >& inputs,
            AbstractNeuron* target, const std::function< double(double) >& func );

    virtual double result() override;
    virtual double target() override;

    void learn();

private:
    std::vector< std::pair< AbstractNeuron*, double > > inputs;
    AbstractNeuron* targetNeuron;
    std::function< double(double) > activation_func;
};

#endif /// NEURON_HPP
