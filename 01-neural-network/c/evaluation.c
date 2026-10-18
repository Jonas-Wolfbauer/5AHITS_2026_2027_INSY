#include "evaluation.h"

#include <stdio.h>

#define XOR_SAMPLES 4
#define SEVEN_SEGMENT_SAMPLES 10
// Largest output count of both datasets
#define MAX_OUTPUTS 7

static const double XOR_INPUTS[XOR_SAMPLES][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const double XOR_TARGETS[XOR_SAMPLES][1] = {{0}, {1}, {1}, {0}};

// 4-bit BCD digit 0-9 -> segments a-g
static const double SEVEN_SEGMENT_INPUTS[SEVEN_SEGMENT_SAMPLES][4] = {
    {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1}, {0, 1, 0, 0},
    {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1}, {1, 0, 0, 0}, {1, 0, 0, 1},
};
static const double SEVEN_SEGMENT_TARGETS[SEVEN_SEGMENT_SAMPLES][7] = {
    {1, 1, 1, 1, 1, 1, 0}, {0, 1, 1, 0, 0, 0, 0}, {1, 1, 0, 1, 1, 0, 1}, {1, 1, 1, 1, 0, 0, 1},
    {0, 1, 1, 0, 0, 1, 1}, {1, 0, 1, 1, 0, 1, 1}, {1, 0, 1, 1, 1, 1, 1}, {1, 1, 1, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 0, 1, 1},
};

void PrintArchitecture (const Network *network)
{
    printf("%d", network->inputSize);
    for (int l = 0; l < network->layerCount; l++)
    {
        printf("-%d", network->layers[l].neuronCount);
    }
}

// Prints one row per sample (input, target, outputs) and returns the number of wrong samples.
// inputs and targets point to the first element of row-major sample arrays.
static int EvaluateSamples (Network *network, const double *inputs, const double *targets,
                            int sampleCount)
{
    int inputCount = network->inputSize;
    int outputCount = NetworkOutputCount(network);
    // One output (XOR) gets more digits than seven segments, as in the Python and C# CLI
    int decimals = outputCount == 1 ? 4 : 2;

    printf("#  input  target   outputs\n");
    int wrong = 0;
    for (int s = 0; s < sampleCount; s++)
    {
        const double *sample = &inputs[s * inputCount];
        const double *target = &targets[s * outputCount];
        double outputs[MAX_OUTPUTS];
        NetworkPredict(network, sample, outputs);

        printf("%d  ", s);
        for (int i = 0; i < inputCount; i++)
        {
            printf("%d", (int)sample[i]);
        }
        printf("%*s", 7 - inputCount, "");
        for (int k = 0; k < outputCount; k++)
        {
            printf("%d", (int)target[k]);
        }
        printf("%*s", 9 - outputCount, "");

        int correct = 1;
        for (int k = 0; k < outputCount; k++)
        {
            // Rounded output: 1 from 0.5 on
            correct &= (outputs[k] >= 0.5) == (target[k] == 1.0);
            printf("%.*f ", decimals, outputs[k]);
        }
        printf(" %s\n", correct ? "ok" : "WRONG");
        wrong += !correct;
    }

    printf("\n%d/%d samples correct\n", sampleCount - wrong, sampleCount);
    return wrong;
}

int EvaluateNetwork (Network *network)
{
    int outputCount = NetworkOutputCount(network);
    if (network->inputSize == 2 && outputCount == 1)
    {
        return EvaluateSamples(network, &XOR_INPUTS[0][0], &XOR_TARGETS[0][0], XOR_SAMPLES);
    }
    if (network->inputSize == 4 && outputCount == 7)
    {
        return EvaluateSamples(network, &SEVEN_SEGMENT_INPUTS[0][0], &SEVEN_SEGMENT_TARGETS[0][0],
                               SEVEN_SEGMENT_SAMPLES);
    }

    fprintf(stderr, "no dataset for this network\n");
    return -1;
}
