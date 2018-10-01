#include "target_neuron.hpp"

TargetNeuron::TargetNeuron( double value )
{
    this->value = value;
}

void TargetNeuron::set( double value )
{
    this->value = value;
}

double TargetNeuron::target()
{
    return value;
}

double TargetNeuron::result()
{
    /// Output neuron has no result.
    return 0.0;
}
