#include "input_neuron.hpp"

InputNeuron::InputNeuron( double value )
{
    this->value = value;
}

void InputNeuron::set( double value )
{
    this->value = value;
}

double InputNeuron::getResult()
{
    return value;
}

double InputNeuron::getTarget()
{
    return 0.0;
}
