// Built-in checks: sigmoid, forward pass against the formula, and that a network loaded from
// .txt and one built from the header compute bit-identical outputs.
#include "neural_network.h"
#include "seven_segment.h"
#include "xor.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

// Longest model path, e.g. "../models/python/seven_segment.txt"
#define PATH_SIZE 512

static int failures = 0;

// Prints one result line; returns condition so callers can stop early.
static int Check (int condition, const char *name)
{
    printf("%s  %s\n", condition ? "ok  " : "FAIL", name);
    failures += !condition;
    return condition;
}

static void CheckSigmoid (void)
{
    int stable = Sigmoid(0.0) == 0.5 && Sigmoid(1000.0) == 1.0 && Sigmoid(-1000.0) == 0.0;
    Check(stable, "sigmoid is stable for extreme inputs");
}

static void CheckForwardPass (void)
{
    // 2-2-1 network with hand-picked parameters
    static const double hiddenWeights[2][2] = {{0.5, -1.0}, {1.5, 0.25}};
    static const double outputWeights[2] = {2.0, -0.75};
    Neuron hidden[2] = {
        {.weights = hiddenWeights[0], .bias = 0.1, .inputCount = 2},
        {.weights = hiddenWeights[1], .bias = -0.2, .inputCount = 2},
    };
    Neuron output[1] = {{.weights = outputWeights, .bias = 0.3, .inputCount = 2}};
    Layer layers[2] = {
        {.neurons = hidden, .neuronCount = 2, .inputCount = 2},
        {.neurons = output, .neuronCount = 1, .inputCount = 2},
    };
    Network network = {.layers = layers, .layerCount = 2, .inputSize = 2};

    const double inputs[2] = {0.3, -0.8};
    double predicted = 0.0;
    NetworkPredict(&network, inputs, &predicted);

    double h1 = 1.0 / (1.0 + exp(-(0.5 * 0.3 - 1.0 * -0.8 + 0.1)));
    double h2 = 1.0 / (1.0 + exp(-(1.5 * 0.3 + 0.25 * -0.8 - 0.2)));
    double expected = 1.0 / (1.0 + exp(-(2.0 * h1 - 0.75 * h2 + 0.3)));
    Check(fabs(predicted - expected) <= 1e-15, "forward pass matches manual computation");
}

static void CheckBrokenFileIsRejected (const char *directory)
{
    // A header is no .txt model, so loading it must fail cleanly
    char path[PATH_SIZE];
    snprintf(path, sizeof path, "%s/xor.h", directory);
    Network network;
    fprintf(stderr, "(expected error) ");
    Check(NetworkLoadTxt(&network, path) == -1, "wrong file format is rejected");
}

// Compares a network loaded from .txt with one built by init for every possible 0/1 input.
static void CheckSameOutputs (const char *directory, const char *name, void (*init)(Network *))
{
    char path[PATH_SIZE];
    char description[PATH_SIZE];
    snprintf(path, sizeof path, "%s/%s.txt", directory, name);
    snprintf(description, sizeof description, "%s.txt and %s.h give bit-identical outputs", name,
             name);

    Network loaded;
    if (!Check(NetworkLoadTxt(&loaded, path) == 0, path))
    {
        return;
    }
    Network built;
    init(&built);

    int inputCount = loaded.inputSize;
    int outputCount = NetworkOutputCount(&loaded);
    int identical = inputCount == built.inputSize && outputCount == NetworkOutputCount(&built);
    for (int pattern = 0; identical && pattern < 1 << inputCount; pattern++)
    {
        double inputs[16];
        double fromTxt[16];
        double fromHeader[16];
        for (int i = 0; i < inputCount; i++)
        {
            inputs[i] = (pattern >> (inputCount - 1 - i)) & 1;
        }
        NetworkPredict(&loaded, inputs, fromTxt);
        NetworkPredict(&built, inputs, fromHeader);
        identical = memcmp(fromTxt, fromHeader, (size_t)outputCount * sizeof(double)) == 0;
    }

    Check(identical, description);
    NetworkFree(&loaded);
}

int main (int argc, char **argv)
{
    // Unbuffered, so the expected error message on stderr appears in the right place
    setvbuf(stdout, NULL, _IONBF, 0);

    // The headers are compiled in; the .txt files must come from the same folder
    const char *directory = argc > 1 ? argv[1] : "../models/python";

    CheckSigmoid();
    CheckForwardPass();
    CheckBrokenFileIsRejected(directory);
    CheckSameOutputs(directory, "xor", XOR_Init);
    CheckSameOutputs(directory, "seven_segment", SEVEN_SEGMENT_Init);

    printf("\n%s\n", failures == 0 ? "all checks passed" : "some checks FAILED");
    return failures == 0 ? 0 : 1;
}
