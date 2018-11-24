#include <cmath>

#include "hamming_neural_network.hpp"

const double HammingNeuralNetwork::epsilon = 0.1;

HammingNeuralNetwork::HammingNeuralNetwork(
        const size_t imageLinearSize, const size_t nNeurons )
    : INeuralNetwork()
    , nNeurons( nNeurons )
    , inputSize( imageLinearSize )
{}

void HammingNeuralNetwork::addSampleToLearningDataSet(
        const std::vector< double >& input,
        const std::vector< double >& target )
{
    if( input.size() != this->inputSize )
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

std::vector< double > HammingNeuralNetwork::recognizeSample( const std::vector< double >& input )
{
    std::vector< double > neuronus;

    for( size_t i = 0; i < samplesMatrix.size(); ++i )
    {
        double sum = 0.0;

        for( size_t j = 0; j < inputSize; ++j )
        {
            sum += weightsMatrix[ i ][ j ] * input[ j ];
        }

        neuronus.push_back( sum + randomShittyParameter );
    }

    std::vector< double > output( neuronus );

    static const int32_t MAX_ITERATIONS = 32;
    for( int32_t iteration = 0; iteration < MAX_ITERATIONS; ++iteration )
    {
        auto prev_output(output);

        for( size_t i = 0; i < neuronus.size(); ++i )
        {
            double sum = 0.0;

            for( size_t j = 0; j < neuronus.size(); ++j )
            {
                if( i == j ) continue;
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
    return this->inputSize;
}

void HammingNeuralNetwork::updateWeightsMatrix()
{
    weightsMatrix.clear();

    for( const auto& sample : samplesMatrix )
    {
        std::vector< double > new_weights;

        for( const auto& sample_data_value : sample.second )
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
        std::vector< double > new_feedbacks;

        for( size_t j = 0; j < samplesMatrix.size(); ++j )
        {
            new_feedbacks.push_back(
                i == j ? 1.0 : -1.0 / samplesMatrix.size() );
        }

        feedbackMatrix.push_back( new_feedbacks );
    }
}

double HammingNeuralNetwork::activation(double arg) const
{
    if( arg <= 0.0 )
        return 0.0;
    else if( 0 < arg && arg <= this->randomShittyParameter )
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

void HammingNeuralNetwork::clear()
{
    samplesMatrix.clear();
    for( size_t i = 0; i < this->nNeurons; ++i )
    {
        std::fill( this->weightsMatrix[ i ].begin(),
                   this->weightsMatrix[ i ].end(), 0.0 );
        std::fill( this->feedbackMatrix[ i ].begin(),
                   this->feedbackMatrix[ i ].end(), 0.0 );
    }
}

std::vector< INeuralNetwork::Matrix > HammingNeuralNetwork::getWeightsMatrices() const
{
    double** weights = new double*[ this->nNeurons ];
    for( size_t i = 0; i < this->nNeurons; i++ )
    {
        weights[ i ] = new double[ this->inputSize ];
        std::copy( this->weightsMatrix[ i ].begin(),
                   this->weightsMatrix[ i ].end(),
                   weights[ i ] );
    }

    double** feedbacks = new double*[ this->nNeurons ];
    for( size_t i = 0; i < this->nNeurons; i++ )
    {
        feedbacks[ i ] = new double[ this->inputSize ];
        std::copy( this->feedbackMatrix[ i ].begin(),
                   this->feedbackMatrix[ i ].end(),
                   feedbacks[ i ] );
    }

    return { { weights, this->nNeurons, this->inputSize },
             { feedbacks, this->nNeurons, this->inputSize } };
}
