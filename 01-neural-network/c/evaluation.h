// Assignment datasets and a printed check of a network against them.
#ifndef EVALUATION_H
#define EVALUATION_H

#include "neural_network.h"

/// Prints the layer sizes, e.g. "4-16-7".
void PrintArchitecture (const Network *network);

/// Runs the network on its dataset (XOR for 2-1, 7-segment decoder for 4-7), prints one row
/// per sample and returns the number of wrong samples, or -1 if no dataset fits the network.
int EvaluateNetwork (Network *network);

#endif // EVALUATION_H
