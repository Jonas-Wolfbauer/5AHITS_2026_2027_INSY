// Part 5: builds both networks from the generated headers and runs them on their datasets.
// No file is read and no heap memory is used, as on a microcontroller.
#include "evaluation.h"
#include "neural_network.h"
#include "seven_segment.h"
#include "xor.h"

#include <stdio.h>

// Builds one network with its init function and evaluates it; returns the number of wrong samples.
static int Run (const char *header, void (*init)(Network *))
{
    Network network;
    init(&network);

    printf("initialized from %s: ", header);
    PrintArchitecture(&network);
    printf("\n\n");
    return EvaluateNetwork(&network);
}

int main (void)
{
    int wrong = Run("xor.h", XOR_Init);
    printf("\n");
    wrong += Run("seven_segment.h", SEVEN_SEGMENT_Init);
    return wrong == 0 ? 0 : 1;
}
