#include <exception>

#include <QDebug>
#include <QtMath>
#include <QString>

#include "hamming_neural_network.hpp"

const double HammingNeuralNetwork::epsilon = 0.1;

HammingNeuralNetwork::HammingNeuralNetwork()
{
    this->samplesMatrix = new QVector<QPair<QString, QVector<double>*>*>();
    this->weightsMatrix = new QVector<QVector<double>*>();
    this->feedbackMatrix = new QVector<QVector<double>*>();
}

void HammingNeuralNetwork::append(const MarkedDrawer& image)
{
    const auto mark = image.getMark();

    QVector<double>* data = new QVector<double>();
    /*
    for (const auto& pixel : image.getPixelData())
    {
        qDebug() << QString::fromWCharArray(L"Pixel value = ") << QString::number(pixel.value());
        data->append(pixel.value() > 0 ? -1.0 : 1.0);
    }
    */

    if (data->size() != this->imageLinearSize)
    {
        QString message = QString::fromWCharArray(L"Image linear size doesn't match one set.");
        throw std::logic_error(message.toStdString());
    }

    auto new_sample = new QPair<QString, QVector<double>*>();
    new_sample->first = std::move(mark);
    new_sample->second = std::move(data);
    this->samplesMatrix->append(new_sample);

    this->updateWeightsMatrix();
    this->updateFeedbackMatrix();
}

QString HammingNeuralNetwork::recognize(const MarkedDrawer& image)
{
    QVector<double>* data = new QVector<double>();
    /*
    for (const auto& pixel : image.getPixelData())
    {
        qDebug() << QString::fromWCharArray(L"Pixel value = ") << QString::number(pixel.value());
        data->append(pixel.value() > 0 ? -1.0 : 1.0);
    }
    */

    QVector<double> neuronus;

    for (qint32 i = 0; i < this->samplesMatrix->size(); ++i)
    {
        double sum = 0.0;

        for (qint32 j = 0; j < this->imageLinearSize; ++j)
        {
            sum += weightsMatrix->at(i)->at(j) * data->at(j);
        }

        neuronus.push_back(sum + this->randomShittyParameter);
    }

    QVector<double> output(neuronus);

    static const qint32 MAX_ITERATIONS = 32;
    for (qint32 iteration = 0; iteration < MAX_ITERATIONS; ++iteration)
    {
        auto prev_output(output);

        for (qint32 i = 0; i < neuronus.size(); ++i)
        {
            double sum = 0.0;

            for (qint32 j = 0; j < neuronus.size(); ++j)
            {
                if (i == j) continue;
                sum += feedbackMatrix->at(i)->at(j) * output[j];
            }

            neuronus[i] = prev_output[i] + sum;
        }

        output = neuronus;
        for (auto& value : output)
        {
            value = this->activation(value);
        }

        QVector<double> temp;
        for (qint32 i = 0; i < output.size(); ++i)
        {
            temp.push_back(output[i] - prev_output[i]);
        }

        if (this->norm(temp) < this->epsilon)
        {
            break;
        }
    }

    QVector<qint32> indices;
    for (qint32 i = 0; i < output.size(); ++i)
    {
        if (output[i] > 0.0)
        {
            indices.push_back(i);
        }
    }

    if (indices.size() == 0)
    {
        return QString();
    }
    else if (indices.size() == 1)
    {
        return samplesMatrix->at(indices[0])->first;
    }
    else
    {
        QString result;
        for (const auto& idx : indices)
        {
            result += samplesMatrix->at(idx)->first + "; ";
        }

        return result;
    }
}

qint32 HammingNeuralNetwork::getImageLinearSize() const
{
    return this->imageLinearSize;
}

void HammingNeuralNetwork::setLinearSize(const qint32 imageLinearSize)
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
    this->setLinearSize(this->imageSize.width() * this->imageSize.height());
}

void HammingNeuralNetwork::updateWeightsMatrix()
{
    weightsMatrix->clear();

    for (const auto& sample : *samplesMatrix)
    {
        QVector<double>* new_weights = new QVector<double>();

        for (const auto& sample_data_value : *sample->second)
        {
            new_weights->append(0.5 * sample_data_value);
        }

        weightsMatrix->append(new_weights);
    }
}

void HammingNeuralNetwork::updateFeedbackMatrix()
{
    this->feedbackMatrix->clear();

    for (qint32 i = 0; i < samplesMatrix->size(); ++i)
    {
        this->feedbackMatrix->append(new QVector<double>());

        for (qint32 j = 0; j < samplesMatrix->size(); ++j)
        {
            this->feedbackMatrix->at(i)->append(
                i == j ? 1.0 : -1.0 / this->samplesMatrix->size()
            );
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
    double result_squared(0.0);

    for(const auto& value : vector)
    {
        result_squared += qPow(value, 2.0);
    }

    return qSqrt(result_squared);
}
