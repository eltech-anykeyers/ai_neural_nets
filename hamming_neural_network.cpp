#include <exception>

#include <QDebug>
#include <QtMath>
#include <QString>

#include "hamming_neural_network.hpp"

const double HammingNeuralNetwork::epsilon = 0.1;

HammingNeuralNetwork::HammingNeuralNetwork()
{}

void HammingNeuralNetwork::addSampleToLearningDataSet(
        const QVector< double >& input,
        const QVector< double >& target )
{
    if( input.size() != this->imageLinearSize )
    {
        throw std::logic_error( "Image linear size doesn't match one set." );
    }

    samplesMatrix.append(qMakePair( target, input ) );
}

void HammingNeuralNetwork::adjustConnectionsWeights()
{
    this->updateWeightsMatrix();
    this->updateFeedbackMatrix();
}

QVector< double > HammingNeuralNetwork::recognizeSample( const QVector< double >& input )
{
    QVector<double> neuronus;

    for ( qint32 i = 0; i < samplesMatrix.size(); ++i )
    {
        double sum = 0.0;

        for ( qint32 j = 0; j < imageLinearSize; ++j )
        {
            sum += weightsMatrix[ i ][ j ] * input[ j ];
        }

        neuronus.push_back( sum + randomShittyParameter );
    }

    QVector<double> output(neuronus);

    static const qint32 MAX_ITERATIONS = 32;
    for( qint32 iteration = 0; iteration < MAX_ITERATIONS; ++iteration )
    {
        auto prev_output(output);

        for( qint32 i = 0; i < neuronus.size(); ++i )
        {
            double sum = 0.0;

            for (qint32 j = 0; j < neuronus.size(); ++j)
            {
                if (i == j) continue;
                sum += feedbackMatrix[ i ][ j ] * output[ j ];
            }

            neuronus[i] = prev_output[i] + sum;
        }

        output = neuronus;
        for( auto& value : output )
        {
            value = this->activation(value);
        }

        QVector<double> temp;
        for( qint32 i = 0; i < output.size(); ++i )
        {
            temp.push_back(output[i] - prev_output[i]);
        }

        if( this->norm(temp) < this->epsilon )
        {
            break;
        }
    }

    return output;
}

qint32 HammingNeuralNetwork::getImageLinearSize() const
{
    return this->imageLinearSize;
}

void HammingNeuralNetwork::setLinearSize( const qint32 imageLinearSize )
{
    this->imageLinearSize = imageLinearSize;
    this->randomShittyParameter = this->imageLinearSize / 2.0;
}

QSize HammingNeuralNetwork::getImageSize() const
{
    return this->imageSize;
}

void HammingNeuralNetwork::setImageSize(const QSize& imageSize)
{
    this->imageSize = imageSize;
    this->setLinearSize( imageSize.width() * imageSize.height() );
}

void HammingNeuralNetwork::updateWeightsMatrix()
{
    weightsMatrix.clear();

    for( const auto& sample : samplesMatrix )
    {
        QVector<double> new_weights;

        for (const auto& sample_data_value : sample.second)
        {
            new_weights.append( 0.5 * sample_data_value );
        }

        weightsMatrix.append( new_weights );
    }
}

void HammingNeuralNetwork::updateFeedbackMatrix()
{
    feedbackMatrix.clear();

    for( qint32 i = 0; i < samplesMatrix.size(); ++i )
    {
        feedbackMatrix.append( QVector< double >() );
        for( qint32 j = 0; j < samplesMatrix.size(); ++j )
        {
            feedbackMatrix[ i ].append(
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

double HammingNeuralNetwork::norm(const QVector<double>& vector) const
{
    double result_squared( 0.0 );

    for(const auto& value : vector)
    {
        result_squared += qPow( value, 2.0 );
    }

    return qSqrt(result_squared);
}
