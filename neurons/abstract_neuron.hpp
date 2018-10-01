#ifndef ABSTRACT_NEURON_HPP
#define ABSTRACT_NEURON_HPP


class AbstractNeuron
{
public:
    AbstractNeuron();
    virtual ~AbstractNeuron() = default;

    virtual double result() = 0;
    virtual double target() = 0;
};

#endif /// ABSTRACT_NEURON_HPP
