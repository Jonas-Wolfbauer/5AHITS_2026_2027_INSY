// Forward-only neural network in C (parts 4 and 5 of the assignment).
// Same design as the Python and C# versions: the input is no layer of its own,
// every layer is fully connected, every neuron uses the sigmoid.
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

/// Single neuron computing a = sigmoid(w . x + b).
typedef struct
{
    // One weight per input
    const double *weights;
    // Bias term b
    double bias;
    // Number of inputs, i.e. neurons in the previous layer
    int inputCount;
    // Output a of the last forward pass
    double output;
} Neuron;

/// Fully connected layer: every neuron sees every input.
typedef struct
{
    Neuron *neurons;
    int neuronCount;
    // Number of inputs, i.e. neurons in the previous layer
    int inputCount;
} Layer;

/// Network from the first hidden layer to the output layer.
typedef struct
{
    Layer *layers;
    int layerCount;
    int inputSize;
} Network;

/// Loads a network from a .txt file written by the Python or C# version.
/// Returns 0 on success; on failure prints the reason to stderr and returns -1.
int NetworkLoadTxt (Network *network, const char *path);

/// Releases a network loaded with NetworkLoadTxt; not for networks built from a header.
void NetworkFree (Network *network);

/// Propagates inputSize values through all layers; every neuron keeps its output.
void NetworkForward (Network *network, const double *inputs);

/// Runs a forward pass and copies the output layer's activations to outputs.
void NetworkPredict (Network *network, const double *inputs, double *outputs);

/// Returns the number of network outputs, i.e. neurons in the output layer.
int NetworkOutputCount (const Network *network);

/// Logistic sigmoid; maps any z into the open interval (0, 1).
double Sigmoid (double z);

#endif // NEURAL_NETWORK_H
