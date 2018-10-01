#include "neuron.hpp"

Neuron::Neuron( const std::vector< AbstractNeuron* >& inputs,
                const std::function< double(double) >& func )
{
    for( const auto& input : inputs )
    {
        this->inputs.push_back( std::make_pair( input, 0.0 ) );
    }
    activation_func = func;
}

double Neuron::getResult()
{
    double result = 0.0;
    for( auto& input : inputs )
    {
        result += input.first->getResult() * input.second;
    }
    return activation_func( result );
}

double Neuron::getTarget()
{
    return 0.0;
}
