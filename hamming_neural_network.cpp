#include <cmath>

#include "hamming_neural_network.hpp"

const double HammingNeuralNetwork::epsilon = 0.1;

HammingNeuralNetwork::HammingNeuralNetwork()
{}

void HammingNeuralNetwork::addSampleToLearningDataSet(
        const std::vector< double >& input,
        const std::vector< double >& target )
{
    if( input.size() != this->imageLinearSize )
    {
        return;
    }

    samplesMatrix.push_back( std::make_pair( target, input ) );
}

void HammingNeuralNetwork::adjustConnectionsWeights()
{
    this->updateWeightsMatrix();
    this->updateFeedbackMatrix();
}

std::vector< double > HammingNeuralNetwork::recognizeSample( const std::vector<  double >& input )
{
    std::vector< double > neuronus;

    for ( size_t i = 0; i < samplesMatrix.size(); ++i )
    {
        double sum = 0.0;

        for ( size_t j = 0; j < imageLinearSize; ++j )
        {
            sum += weightsMatrix[ i ][ j ] * input[ j ];
        }

        neuronus.push_back( sum + randomShittyParameter );
    }

    std::vector< double > output(neuronus);

    static const int32_t MAX_ITERATIONS = 32;
    for( int32_t iteration = 0; iteration < MAX_ITERATIONS; ++iteration )
    {
        auto prev_output(output);

        for( size_t i = 0; i < neuronus.size(); ++i )
        {
            double sum = 0.0;

            for( size_t j = 0; j < neuronus.size(); ++j )
            {
                if (i == j) continue;
                sum += feedbackMatrix[ i ][ j ] * output[ j ];
            }

            neuronus[ i ] = prev_output[ i ] + sum;
        }

        output = neuronus;
        for( auto& value : output )
        {
            value = this->activation(value);
        }

        std::vector< double > temp;
        for( size_t i = 0; i < output.size(); ++i )
        {
            temp.push_back( output[ i ] - prev_output[ i ] );
        }

        if( this->norm(temp) < this->epsilon )
        {
            break;
        }
    }

    return output;
}

size_t HammingNeuralNetwork::getImageLinearSize() const
{
    return this->imageLinearSize;
}

void HammingNeuralNetwork::setLinearSize( const size_t imageLinearSize )
{
    this->imageLinearSize = imageLinearSize;
    this->randomShittyParameter = this->imageLinearSize / 2.0;
}

void HammingNeuralNetwork::updateWeightsMatrix()
{
    weightsMatrix.clear();

    for( const auto& sample : samplesMatrix )
    {
        std::vector< double > new_weights;

        for (const auto& sample_data_value : sample.second)
        {
            new_weights.push_back( 0.5 * sample_data_value );
        }

        weightsMatrix.push_back( new_weights );
    }
}

void HammingNeuralNetwork::updateFeedbackMatrix()
{
    feedbackMatrix.clear();

    for( size_t i = 0; i < samplesMatrix.size(); ++i )
    {
        feedbackMatrix.push_back( std::vector< double >() );
        for( size_t j = 0; j < samplesMatrix.size(); ++j )
        {
            feedbackMatrix[ i ].push_back(
                i == j ? 1.0 : -1.0 / samplesMatrix.size() );
        }
    }
}

double HammingNeuralNetwork::activation(double arg) const
{
    if (arg <= 0.0)
        return 0.0;
    else if (0 < arg && arg <= this->randomShittyParameter)
        return arg;
    else
        return this->randomShittyParameter;
}

double HammingNeuralNetwork::norm(const std::vector< double >& vector) const
{
    return std::sqrt(
        std::accumulate( vector.begin(), vector.end(), 0.0,
                         []( double acc, double val ){
                             return acc + std::pow( val, 2.0 );
                         } ) );
}
