#include "neuron.hpp"

Neuron::Neuron( const std::vector< AbstractNeuron* >& inputs,
                AbstractNeuron* target, const std::function< double(double) >& func )
{
    for( const auto& input : inputs )
    {
        this->inputs.push_back( std::make_pair( input, 0.0 ) );
    }
    targetNeuron = target;
    activation_func = func;
}

double Neuron::result()
{
    double result = 0.0;
    for( auto& input : inputs )
    {
        result += input.first->result() * input.second;
    }
    return activation_func( result );
}

double Neuron::target()
{
    return targetNeuron->target();
}

void Neuron::learn()
{
    for( auto& input : inputs )
    {
        input.second += input.first->result() * targetNeuron->target();
    }
}
