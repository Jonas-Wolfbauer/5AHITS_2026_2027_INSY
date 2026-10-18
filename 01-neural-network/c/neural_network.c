#include "neural_network.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Room for the activation name in a file; the "%31s" below must stay one shorter
#define ACTIVATION_NAME_SIZE 32

// Prints why loading failed, releases everything loaded so far and returns -1.
static int LoadFailed (Network *network, FILE *file, const char *path, const char *reason)
{
    fprintf(stderr, "%s: %s\n", path, reason);
    fclose(file);
    NetworkFree(network);
    return -1;
}

int NetworkLoadTxt (Network *network, const char *path)
{
    *network = (Network){0};

    FILE *file = fopen(path, "r");
    if (file == NULL)
    {
        fprintf(stderr, "%s: cannot open file\n", path);
        return -1;
    }

    // Every value is preceded by its keyword, so one format string reads the whole header
    int version = 0;
    int layerCount = 0;
    char activation[ACTIVATION_NAME_SIZE] = "";
    int fields = fscanf(file, " insy-nn %d activation %31s input_size %d layers %d", &version,
                        activation, &network->inputSize, &layerCount);
    if (fields != 4 || version != 1)
    {
        return LoadFailed(network, file, path, "not an insy-nn v1 network");
    }

    if (strcmp(activation, "sigmoid") != 0)
    {
        return LoadFailed(network, file, path, "only the sigmoid activation is supported");
    }

    if (network->inputSize < 1 || layerCount < 1)
    {
        return LoadFailed(network, file, path, "invalid sizes");
    }

    // Zeroed memory, so NetworkFree can clean up a partly loaded network
    network->layers = calloc((size_t)layerCount, sizeof(Layer));
    if (network->layers == NULL)
    {
        return LoadFailed(network, file, path, "out of memory");
    }
    network->layerCount = layerCount;

    int expectedInputs = network->inputSize;
    for (int l = 0; l < layerCount; l++)
    {
        Layer *layer = &network->layers[l];
        int number = 0;
        fields = fscanf(file, " layer %d neurons %d inputs %d", &number, &layer->neuronCount,
                        &layer->inputCount);
        if (fields != 3 || number != l + 1 || layer->neuronCount < 1 ||
            layer->inputCount != expectedInputs)
        {
            layer->neuronCount = 0;
            return LoadFailed(network, file, path, "unexpected layer header");
        }

        layer->neurons = calloc((size_t)layer->neuronCount, sizeof(Neuron));
        if (layer->neurons == NULL)
        {
            layer->neuronCount = 0;
            return LoadFailed(network, file, path, "out of memory");
        }

        for (int n = 0; n < layer->neuronCount; n++)
        {
            Neuron *neuron = &layer->neurons[n];
            double *weights = malloc((size_t)layer->inputCount * sizeof(double));
            if (weights == NULL)
            {
                return LoadFailed(network, file, path, "out of memory");
            }
            neuron->weights = weights;
            neuron->inputCount = layer->inputCount;

            fields = fscanf(file, " neuron %d bias %lf weights", &number, &neuron->bias);
            if (fields != 2 || number != n + 1)
            {
                return LoadFailed(network, file, path, "unexpected neuron line");
            }

            for (int w = 0; w < neuron->inputCount; w++)
            {
                if (fscanf(file, "%lf", &weights[w]) != 1)
                {
                    return LoadFailed(network, file, path, "missing weight");
                }
            }
        }

        expectedInputs = layer->neuronCount;
    }

    fclose(file);
    return 0;
}

void NetworkFree (Network *network)
{
    if (network->layers != NULL)
    {
        for (int l = 0; l < network->layerCount; l++)
        {
            Layer *layer = &network->layers[l];
            if (layer->neurons == NULL)
            {
                continue;
            }

            for (int n = 0; n < layer->neuronCount; n++)
            {
                // Allocated by NetworkLoadTxt, so freeing through the const pointer is fine
                free((void *)layer->neurons[n].weights);
            }
            free(layer->neurons);
        }
        free(network->layers);
    }

    *network = (Network){0};
}

void NetworkForward (Network *network, const double *inputs)
{
    for (int l = 0; l < network->layerCount; l++)
    {
        Layer *layer = &network->layers[l];

        // The first layer reads the network inputs, every other layer the outputs before it
        const Layer *previous = l > 0 ? &network->layers[l - 1] : NULL;

        for (int n = 0; n < layer->neuronCount; n++)
        {
            Neuron *neuron = &layer->neurons[n];
            double sum = 0.0;
            for (int i = 0; i < neuron->inputCount; i++)
            {
                double input = previous != NULL ? previous->neurons[i].output : inputs[i];
                sum += neuron->weights[i] * input;
            }
            neuron->output = Sigmoid(sum + neuron->bias);
        }
    }
}

void NetworkPredict (Network *network, const double *inputs, double *outputs)
{
    NetworkForward(network, inputs);

    const Layer *last = &network->layers[network->layerCount - 1];
    for (int n = 0; n < last->neuronCount; n++)
    {
        outputs[n] = last->neurons[n].output;
    }
}

int NetworkOutputCount (const Network *network)
{
    return network->layers[network->layerCount - 1].neuronCount;
}

double Sigmoid (double z)
{
    // Same two branches as the Python and C# versions: exp only ever gets a non-positive argument
    if (z >= 0.0)
    {
        return 1.0 / (1.0 + exp(-z));
    }

    double expZ = exp(z);
    return expZ / (1.0 + expZ);
}
