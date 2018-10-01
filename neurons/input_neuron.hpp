#ifndef INPUT_NEURON_HPP
#define INPUT_NEURON_HPP

#include "abstract_neuron.hpp"

class InputNeuron : public AbstractNeuron
{
public:
    InputNeuron( double value );

    void set( double value );
    virtual double getResult() override;
    virtual double getTarget() override;

private:
    double value;
};

#endif /// INPUT_NEURON_HPP
