#ifndef ABSTRACT_NEURON_HPP
#define ABSTRACT_NEURON_HPP


class AbstractNeuron
{
public:
    AbstractNeuron();
    virtual ~AbstractNeuron() = default;

    virtual double getResult() = 0;
    virtual double getTarget() = 0;
};

#endif /// ABSTRACT_NEURON_HPP
