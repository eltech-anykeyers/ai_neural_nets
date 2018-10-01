#ifndef TARGET_NEURON_HPP
#define TARGET_NEURON_HPP

#include "abstract_neuron.hpp"

class TargetNeuron : public AbstractNeuron
{
public:
    TargetNeuron( double value );

    void set( double value );
    virtual double target() override;

private:
    virtual double result() override;

    double value;
};

#endif /// TARGET_NEURON_HPP
