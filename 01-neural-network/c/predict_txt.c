// Part 4: loads a network from a .txt file and runs it on its dataset.
#include "evaluation.h"
#include "neural_network.h"

#include <stdio.h>

int main (int argc, char **argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "usage: %s MODEL.txt\n", argv[0]);
        return 2;
    }

    Network network;
    if (NetworkLoadTxt(&network, argv[1]) != 0)
    {
        return 2;
    }

    printf("loaded %s: ", argv[1]);
    PrintArchitecture(&network);
    printf("\n\n");

    int wrong = EvaluateNetwork(&network);
    NetworkFree(&network);
    return wrong == 0 ? 0 : 1;
}
