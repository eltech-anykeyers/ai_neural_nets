#include <neural_nets/hebbian_neural_network.hpp>

#include <cmath>

HebbianNeuralNetwork::HebbianNeuralNetwork( size_t inputSize, size_t nNeurons  )
{
    this->inputSize = inputSize + 1;
    this->nNeurons = nNeurons;

    inputLayer = std::vector< InputNeuron >( inputSize );
    targetLayer = std::vector< TargetNeuron >( nNeurons );

    std::vector< AbstractNeuron* > inputs;
    std::transform( inputLayer.begin(), inputLayer.end(), std::back_inserter( inputs ),
                    []( InputNeuron& n )-> AbstractNeuron* { return &n; });
    auto binary = []( double s ) -> double { return s > 0 ? 1.0 : 0.0; };
    for( auto& target : targetLayer )
    {
        neuronsLayer.push_back( Neuron( inputs, &target, binary ) );
    }
}

void HebbianNeuralNetwork::addLearningDataSet(
    const std::vector< double >& dataSet, const std::vector< double >& target )
{
    data.push_back( std::make_pair( dataSet, target ) );
}

void HebbianNeuralNetwork::learn()
{
    bool stop = false;
    do
    {
        /// Learn.
        for( const auto& set : data )
        {
            this->setDataSet( set.first, set.second );
            do
            {
                for( auto& neuron : neuronsLayer )
                {
                    neuron.learn();
                }
            }
            while( !this->compare() );
        }

        /// Validate.
        stop = true;
        for( const auto& set : data )
        {
            this->setDataSet( set.first, set.second );
            stop = stop && this->compare();
            if( !stop ) break;
        }
    }
    while( !stop );
}

bool HebbianNeuralNetwork::test(
    const std::vector< double >& dataSet, const std::vector< double >& target )
{
    this->setDataSet( dataSet, target );
    return this->compare();
}

bool HebbianNeuralNetwork::compare()
{
    bool equal = true;
    for( auto& neuron : neuronsLayer )
    {
        equal = equal &&
                ( fabs( neuron.result() - neuron.target() ) < 1e-7 );
        if( !equal ) break;
    }
    return equal;
}

void HebbianNeuralNetwork::setDataSet(
    const std::vector< double >& dataSet, const std::vector< double >& target )
{
    inputLayer[ 0 ].set( 1.0 );
    for( std::vector< double >::size_type i = 1; i <= dataSet.size(); i++ )
    {
        inputLayer[ i ].set( dataSet[ i ] );
    }
    for( std::vector< double >::size_type i = 0; i < target.size(); i++ )
    {
        targetLayer[ i ].set( target[ i ] );
    }
}
