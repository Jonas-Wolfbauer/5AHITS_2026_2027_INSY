namespace NeuralNetworkFromScratch
{
    /// <summary>
    /// All parameters of a network in the layout shared by every file format and by the Python version
    /// (format "insy-nn", version 1).
    /// </summary>
    /// <param name="Activation">Activation of all neurons, e.g. "sigmoid".</param>
    /// <param name="InputSize">Number of network inputs.</param>
    /// <param name="Layers">Layers from the first hidden layer to the output layer.</param>
    public record NetworkData (string Activation, int InputSize, IReadOnlyList<LayerData> Layers);

    /// <summary>One layer: its neurons in order.</summary>
    /// <param name="Neurons">Neurons of the layer.</param>
    public record LayerData (IReadOnlyList<NeuronData> Neurons);

    /// <summary>One neuron: a weight per input and the bias.</summary>
    /// <param name="Weights">One weight per input.</param>
    /// <param name="Bias">Bias term.</param>
    public record NeuronData (double[] Weights, double Bias);
}
