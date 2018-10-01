#include "input_neuron.hpp"

InputNeuron::InputNeuron( double value )
{
    this->value = value;
}

void InputNeuron::set( double value )
{
    this->value = value;
}

double InputNeuron::result()
{
    return value;
}

double InputNeuron::target()
{
    /// Input neuron nas no target.
    return 0.0;
}
