namespace NeuralNetworkFromScratch
{
    /// <summary>Fully connected layer: every neuron sees every input.</summary>
    public class Layer
    {
        /// <summary>Creates a layer of neurons with random weights.</summary>
        /// <param name="inputSize">Number of inputs, i.e. neurons in the previous layer.</param>
        /// <param name="neuronCount">Number of neurons in this layer.</param>
        /// <param name="random">Source for the initial weights.</param>
        /// <param name="activation">Activation of all neurons in this layer.</param>
        public Layer (int inputSize, int neuronCount, Random random, string activation = Activations.SigmoidName)
        {
            InputSize = inputSize;
            Neurons = new Neuron[neuronCount];
            for (int n = 0; n < neuronCount; n++)
            {
                Neurons[n] = new Neuron(inputSize, random, activation);
            }
        }

        /// <summary>Creates a layer from existing neurons, e.g. when loading a file.</summary>
        public Layer (int inputSize, Neuron[] neurons)
        {
            InputSize = inputSize;
            Neurons = neurons;
        }

        /// <summary>Number of inputs, i.e. neurons in the previous layer.</summary>
        public int InputSize { get; }

        /// <summary>Neurons of this layer.</summary>
        public Neuron[] Neurons { get; }

        /// <summary>Number of neurons in this layer.</summary>
        public int NeuronCount => Neurons.Length;

        /// <summary>Computes the output of every neuron.</summary>
        /// <param name="inputs">Outputs of the previous layer.</param>
        /// <returns>One activation per neuron.</returns>
        public double[] Forward (double[] inputs)
        {
            // A new array per call: the next layer's neurons keep a reference to it for backpropagation
            double[] outputs = new double[Neurons.Length];
            for (int n = 0; n < Neurons.Length; n++)
            {
                outputs[n] = Neurons[n].Activate(inputs);
            }

            return outputs;
        }

        /// <summary>Backpropagates through this layer and updates its neurons.</summary>
        /// <param name="errors">dJ/da for every neuron of this layer.</param>
        /// <param name="learningRate">Step size α.</param>
        /// <returns>dJ/da for every neuron of the previous layer.</returns>
        public double[] Backward (double[] errors, double learningRate)
        {
            double[] previousErrors = new double[InputSize];

            for (int n = 0; n < Neurons.Length; n++)
            {
                Neuron neuron = Neurons[n];

                // dJ/dz = dJ/da * da/dz
                double delta = errors[n] * neuron.ActivateDerivative();

                // dz/dx_i = w_i; must use the weights from the forward pass,
                // so this happens before the update below
                for (int i = 0; i < InputSize; i++)
                {
                    previousErrors[i] += delta * neuron.Weights[i];
                }

                neuron.UpdateWeights(delta, learningRate);
                neuron.UpdateBias(delta, learningRate);
            }

            return previousErrors;
        }
    }
}
