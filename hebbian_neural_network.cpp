#include "hebbian_neural_network.hpp"

#include <cmath>

HebbianNeuralNetwork::HebbianNeuralNetwork( size_t inputSize, size_t nNeurons  )
    : inputSize( inputSize + 1 )
    , nNeurons( nNeurons )
{
    /// Allocate memory for connections.
    connectionsWeightsMatrix = new double*[ this->nNeurons ];
    for( size_t i = 0; i < this->nNeurons; i++ )
    {
        connectionsWeightsMatrix[ i ] = new double[ this->inputSize ];
        std::fill( connectionsWeightsMatrix[ i ], connectionsWeightsMatrix[ i ] + this->inputSize, 0.0 );
    }

    activation_func =
        []( double s ) -> double
        {
            return s > 0.0 ? 1.0 : 0.0;
        };
}

HebbianNeuralNetwork::HebbianNeuralNetwork(
        size_t weightsMatrixWidth, size_t weightsMatrixHeight,
        double** weightsMatrix )
    : HebbianNeuralNetwork( weightsMatrixHeight - 1, weightsMatrixWidth )
{
    for( size_t i = 0; i < weightsMatrixWidth; i++ )
    {
        std::copy( weightsMatrix[ i ], weightsMatrix[ i ] + weightsMatrixHeight,
                   this->connectionsWeightsMatrix[ i ] );
        delete[] weightsMatrix[ i ];
    }
    delete[] weightsMatrix;
}

HebbianNeuralNetwork::~HebbianNeuralNetwork()
{
    /// Free memory for connections.
    for( size_t i = 0; i < this->nNeurons; i++ )
    {
        delete[] connectionsWeightsMatrix[ i ];
    }
    delete[] connectionsWeightsMatrix;

    /// Free memory for learning data.
    for( const auto& set : data )
    {
        delete[] set.first;
        delete[] set.second;
    }
}

void HebbianNeuralNetwork::addSampleToLearningDataSet(
    const std::vector< double >& input, const std::vector< double >& target )
{
    /// Check sizes.
    if( input.size() + 1 != this->inputSize ) return;
    if( target.size() != this->nNeurons ) return;

    /// Create input array.
    double* inputArray = new double[ this->inputSize ];
    std::copy( input.begin(), input.end(), inputArray + 1 );
    inputArray[ 0 ] = 1.0;

    /// Create target array.
    double* targetArray = new double[ this->nNeurons ];
    std::copy( target.begin(), target.end(), targetArray );

    /// Check if duplicate.
    bool duplicate = false;
    for( const auto& set : data )
    {
        /// Check if input is duplicate.
        duplicate = true;
        for( size_t i = 0; i < this->inputSize; i++ )
        {
            duplicate =
                duplicate &&
                fabs( set.first[ i ] - inputArray[ i ] ) < 1e-7;
            if( !duplicate ) break;
        }
        if( duplicate ) break;

        /// Check if target is duplicate.
        duplicate = true;
        for( size_t i = 0; i < this->nNeurons; i++ )
        {
            duplicate =
                duplicate &&
                fabs( set.second[ i ] - targetArray[ i ] ) < 1e-7;
            if( !duplicate ) break;
        }
        if( duplicate ) break;
    }

    /// Add to learning data.
    if( !duplicate )
    {
        data.push_back( std::make_pair( inputArray, targetArray ) );
    }
}

void HebbianNeuralNetwork::adjustConnectionsWeights()
{
    bool stop = false;
    do
    {
        /// Learn.
        for( const auto& set : data )
        {
            do
            {
                this->adjust( set.first, set.second );
            }
            while( !this->compare( set.first, set.second ) );
        }

        /// Validate.
        stop = true;
        for( const auto& set : data )
        {
            stop = stop && this->compare( set.first, set.second );
            if( !stop ) break;
        }
    }
    while( !stop );
}

std::vector< double > HebbianNeuralNetwork::recognizeSample(
    const std::vector< double >& input )
{
    /// Check sizes.
    if( input.size() + 1 != this->inputSize )
    {
        return std::vector< double >();
    }

    /// Create input array.
    double* inputArray = new double[ this->inputSize ];
    std::copy( input.begin(), input.end(), inputArray + 1 );
    inputArray[ 0 ] = 1.0;

    std::vector< double > result;
    for( size_t i = 0; i < nNeurons; i++ )
    {
        result.push_back( this->compute( i, inputArray ) );
    }

    delete[] inputArray;

    return result;
}

void HebbianNeuralNetwork::clear()
{
    data.clear();
    for( size_t i = 0; i < this->nNeurons; i++ )
    {
        for( size_t j = 0; j < this->inputSize; j++ )
        {
            connectionsWeightsMatrix[ i ][ j ] = 0.0;
        }
    }
}

double** HebbianNeuralNetwork::getWeights() const
{
    double** weights = new double*[ this->nNeurons ];
    for( size_t i = 0; i < this->nNeurons; i++ )
    {
        weights[ i ] = new double[ this->inputSize ];
        std::copy( this->connectionsWeightsMatrix[ i ],
                   this->connectionsWeightsMatrix[ i ] + inputSize,
                   weights[ i ] );
    }
    return weights;
}

double HebbianNeuralNetwork::compute( size_t neuronIndex, double* input )
{
    double result = 0.0;
    for( size_t i = 0; i < inputSize; i++ )
    {
        result += connectionsWeightsMatrix[ neuronIndex ][ i ] * input[ i ];
    }
    return activation_func( result );
}

bool HebbianNeuralNetwork::compare( double* input, double* target )
{
    bool equal = true;
    for( size_t i = 0; i < nNeurons; i++ )
    {
        equal = equal &&
                ( fabs( this->compute( i, input ) - target[ i ] ) < 1e-7 );
        if( !equal ) break;
    }
    return equal;
}

void HebbianNeuralNetwork::adjust( double* input, double* target )
{
    for( size_t i = 0; i < nNeurons; i++ )
    {
        double output = this->compute( i, input );
        if( fabs( output - 1.0 ) < 1e-7 &&
            fabs( target[ i ] ) < 1e-7 )
        {
            /// output > target => '-'
            for( size_t j = 0; j < inputSize; j++ )
            {
                double dw = input[ j ] > 0.0 ? -1.0 : 0.0;
                connectionsWeightsMatrix[ i ][ j ] += dw;
            }
        }
        else if( fabs( output ) < 1e-7 &&
                 fabs( target[ i ]  - 1.0) < 1e-7 )
        {
            /// output < target => '+'
            for( size_t j = 0; j < inputSize; j++ )
            {
                double dw = input[ j ] > 0.0 ? 1.0 : 0.0;
                connectionsWeightsMatrix[ i ][ j ] += dw;
            }
        }
    }
}
