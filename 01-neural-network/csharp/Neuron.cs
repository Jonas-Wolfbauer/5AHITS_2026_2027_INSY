namespace NeuralNetworkFromScratch
{
    /// <summary>
    /// Single neuron computing a = activation(w · x + b).
    /// Stores its last inputs and output, because backpropagation needs both.
    /// </summary>
    public class Neuron
    {
        private readonly Func<double, double> _activationFunction;
        private readonly Func<double, double> _activationDerivative;

        /// <summary>Creates a neuron with random weights and bias in [−1, 1).</summary>
        /// <param name="inputSize">Number of inputs, i.e. neurons in the previous layer.</param>
        /// <param name="random">Source for the initial values; pass a seeded instance for reproducible runs.</param>
        /// <param name="activation">Name of the activation function, see <see cref="Activations"/>.</param>
        public Neuron (int inputSize, Random random, string activation = Activations.SigmoidName)
            : this(new double[inputSize], 0.0, activation)
        {
            // Symmetric range around 0, so neurons start out pulling in both directions
            for (int i = 0; i < inputSize; i++)
            {
                Weights[i] = RandomWeight(random);
            }

            Bias = RandomWeight(random);
        }

        private Neuron (double[] weights, double bias, string activation)
        {
            (_activationFunction, _activationDerivative) = Activations.Get(activation);
            Weights = weights;
            Bias = bias;
            Activation = activation;

            // Unknown until the first forward pass
            Inputs = new double[weights.Length];
            Output = 0.0;
        }

        /// <summary>One weight per input.</summary>
        public double[] Weights { get; }

        /// <summary>Bias term b.</summary>
        public double Bias { get; set; }

        /// <summary>Name of the activation function.</summary>
        public string Activation { get; }

        /// <summary>Inputs x of the last forward pass.</summary>
        public double[] Inputs { get; private set; }

        /// <summary>Output a of the last forward pass.</summary>
        public double Output { get; private set; }

        /// <summary>Creates a neuron with given parameters, e.g. when loading a file.</summary>
        public static Neuron FromParameters (double[] weights, double bias, string activation = Activations.SigmoidName)
        {
            // A copy, so the neuron never shares its weights with the caller
            return new Neuron((double[])weights.Clone(), bias, activation);
        }

        /// <summary>Runs the forward pass of this neuron.</summary>
        /// <param name="inputs">Outputs of the previous layer (or the network input).</param>
        /// <returns>a = activation(z) with z = w · x + b.</returns>
        public double Activate (double[] inputs)
        {
            Inputs = inputs;
            double z = NeuralNetworkMath.ComputeNeuron(inputs, Weights, Bias);
            Output = _activationFunction(z);
            return Output;
        }

        /// <summary>Returns da/dz for the last forward pass.</summary>
        public double ActivateDerivative ()
        {
            return _activationDerivative(Output);
        }

        /// <summary>Takes one gradient-descent step on the weights.</summary>
        /// <param name="delta">dJ/dz of this neuron.</param>
        /// <param name="learningRate">Step size α.</param>
        public void UpdateWeights (double delta, double learningRate)
        {
            for (int i = 0; i < Weights.Length; i++)
            {
                // dJ/dw_i = dJ/dz * dz/dw_i = delta * x_i
                Weights[i] -= learningRate * delta * Inputs[i];
            }
        }

        /// <summary>Takes one gradient-descent step on the bias.</summary>
        /// <param name="delta">dJ/dz of this neuron.</param>
        /// <param name="learningRate">Step size α.</param>
        public void UpdateBias (double delta, double learningRate)
        {
            // dz/db = 1, so dJ/db = delta
            Bias -= learningRate * delta;
        }

        private static double RandomWeight (Random random)
        {
            // NextDouble lies in [0, 1); scaled and shifted to [-1, 1)
            return random.NextDouble() * 2.0 - 1.0;
        }
    }
}
