using System.Text;

namespace NeuralNetworkFromScratch
{
    /// <summary>Feedforward network trained with stochastic gradient descent.</summary>
    public class NeuralNetwork
    {
        /// <summary>Creates a network with random weights.</summary>
        /// <param name="layerSizes">
        /// Input size followed by the neuron count of every layer, e.g. [2, 2, 1] for 2 inputs,
        /// 2 hidden neurons and 1 output. The input is not a layer of its own: it has no weights.
        /// </param>
        /// <param name="learningRate">Step size α for every update.</param>
        /// <param name="seed">Seed for the initial weights; null gives a different network per run.</param>
        /// <param name="activation">Activation of all neurons.</param>
        public NeuralNetwork (
            int[] layerSizes,
            double learningRate = 0.1,
            int? seed = null,
            string activation = Activations.SigmoidName)
        {
            if (layerSizes.Length < 2 || layerSizes.Any(size => size < 1))
            {
                throw new ArgumentException($"invalid layer sizes: [{string.Join(", ", layerSizes)}]", nameof(layerSizes));
            }

            LearningRate = learningRate;
            Activation = activation;

            Random random = seed is null ? new Random() : new Random(seed.Value);
            Layers = new Layer[layerSizes.Length - 1];
            for (int l = 0; l < Layers.Length; l++)
            {
                // Each layer has as many inputs as the layer before it has neurons
                Layers[l] = new Layer(layerSizes[l], layerSizes[l + 1], random, activation);
            }
        }

        private NeuralNetwork (Layer[] layers, double learningRate, string activation)
        {
            Layers = layers;
            LearningRate = learningRate;
            Activation = activation;
        }

        /// <summary>Step size α for every update.</summary>
        public double LearningRate { get; set; }

        /// <summary>Activation of all neurons.</summary>
        public string Activation { get; }

        /// <summary>Layers from the first hidden layer to the output layer.</summary>
        public Layer[] Layers { get; }

        /// <summary>Number of network inputs.</summary>
        public int InputSize => Layers[0].InputSize;

        /// <summary>Input size followed by the neuron count of every layer.</summary>
        public int[] LayerSizes => [InputSize, .. Layers.Select(layer => layer.NeuronCount)];

        /// <summary>Returns the cost J = 1/2 · Σ (ŷ − y)² of one sample.</summary>
        public static double Cost (double[] outputs, double[] targets)
        {
            double squaredErrorSum = 0.0;
            for (int k = 0; k < outputs.Length; k++)
            {
                double difference = outputs[k] - targets[k];
                squaredErrorSum += difference * difference;
            }

            return 0.5 * squaredErrorSum;
        }

        /// <summary>Runs a forward pass.</summary>
        /// <param name="inputs">One sample with <see cref="InputSize"/> values.</param>
        /// <returns>Activations of the output layer.</returns>
        public double[] Predict (double[] inputs)
        {
            if (inputs.Length != InputSize)
            {
                throw new ArgumentException($"expected {InputSize} inputs, got {inputs.Length}", nameof(inputs));
            }

            double[] outputs = inputs;
            foreach (Layer layer in Layers)
            {
                outputs = layer.Forward(outputs);
            }

            return outputs;
        }

        /// <summary>Runs one SGD step: forward pass, backward pass, update.</summary>
        /// <param name="inputs">One sample with <see cref="InputSize"/> values.</param>
        /// <param name="targets">Expected output for this sample.</param>
        /// <returns>Cost J of this sample before the update.</returns>
        public double Train (double[] inputs, double[] targets)
        {
            double[] outputs = Predict(inputs);
            if (targets.Length != outputs.Length)
            {
                throw new ArgumentException($"expected {outputs.Length} targets, got {targets.Length}", nameof(targets));
            }

            // dJ/dy_hat for J = 1/2 * (y_hat - y)^2
            double[] errors = new double[outputs.Length];
            for (int k = 0; k < outputs.Length; k++)
            {
                errors[k] = outputs[k] - targets[k];
            }

            for (int l = Layers.Length - 1; l >= 0; l--)
            {
                errors = Layers[l].Backward(errors, LearningRate);
            }

            return Cost(outputs, targets);
        }

        /// <summary>Trains on a whole dataset for a number of epochs.</summary>
        /// <param name="inputs">All samples.</param>
        /// <param name="targets">Expected output per sample.</param>
        /// <param name="epochs">Passes over the dataset; each pass updates once per sample.</param>
        /// <param name="onEpoch">
        /// Called after every epoch with the number of finished epochs (starting at 1) and the mean cost
        /// of that epoch, e.g. for progress output.
        /// </param>
        /// <returns>Mean cost per epoch.</returns>
        public List<double> Fit (double[][] inputs, double[][] targets, int epochs, Action<int, double>? onEpoch = null)
        {
            if (inputs.Length != targets.Length || inputs.Length == 0)
            {
                throw new ArgumentException("inputs and targets must be non-empty and equally long");
            }

            var losses = new List<double>(epochs);
            for (int epoch = 1; epoch <= epochs; epoch++)
            {
                double epochLoss = 0.0;
                for (int s = 0; s < inputs.Length; s++)
                {
                    epochLoss += Train(inputs[s], targets[s]);
                }

                losses.Add(epochLoss / inputs.Length);
                onEpoch?.Invoke(epoch, losses[^1]);
            }

            return losses;
        }

        /// <summary>Computes the mean cost over a dataset without training.</summary>
        public double MeanCost (double[][] inputs, double[][] targets)
        {
            double total = 0.0;
            for (int s = 0; s < inputs.Length; s++)
            {
                total += Cost(Predict(inputs[s]), targets[s]);
            }

            return total / inputs.Length;
        }

        /// <summary>Returns all parameters in the layout shared by the file formats.</summary>
        public NetworkData ToData ()
        {
            var layers = new List<LayerData>();
            foreach (Layer layer in Layers)
            {
                var neurons = new List<NeuronData>();
                foreach (Neuron neuron in layer.Neurons)
                {
                    // A copy, so further training does not change exported data
                    neurons.Add(new NeuronData((double[])neuron.Weights.Clone(), neuron.Bias));
                }

                layers.Add(new LayerData(neurons));
            }

            return new NetworkData(Activation, InputSize, layers);
        }

        /// <summary>Rebuilds a network with exactly the stored parameters.</summary>
        /// <param name="data">Parameters, e.g. read from a file.</param>
        /// <param name="learningRate">Step size for further training; not part of the files.</param>
        public static NeuralNetwork FromData (NetworkData data, double learningRate = 0.1)
        {
            NetworkSerializer.Validate(data);

            var layers = new Layer[data.Layers.Count];
            int inputSize = data.InputSize;
            for (int l = 0; l < layers.Length; l++)
            {
                IReadOnlyList<NeuronData> neuronData = data.Layers[l].Neurons;
                var neurons = new Neuron[neuronData.Count];
                for (int n = 0; n < neurons.Length; n++)
                {
                    neurons[n] = Neuron.FromParameters(neuronData[n].Weights, neuronData[n].Bias, data.Activation);
                }

                layers[l] = new Layer(inputSize, neurons);
                inputSize = neurons.Length;
            }

            return new NeuralNetwork(layers, learningRate, data.Activation);
        }

        /// <summary>Saves the network as JSON.</summary>
        public void DumpJson (string path)
        {
            WriteText(path, NetworkSerializer.ToJson(ToData()));
        }

        /// <summary>Loads a network saved with <see cref="DumpJson"/> (in C# or Python).</summary>
        public static NeuralNetwork Load (string path)
        {
            return FromData(NetworkSerializer.FromJson(File.ReadAllText(path)));
        }

        /// <summary>Saves the network as human-readable text.</summary>
        public void DumpTxt (string path)
        {
            WriteText(path, NetworkSerializer.ToTxt(ToData()));
        }

        /// <summary>Loads a network saved with <see cref="DumpTxt"/> (in C# or Python).</summary>
        public static NeuralNetwork LoadTxt (string path)
        {
            return FromData(NetworkSerializer.FromTxt(File.ReadAllText(path)));
        }

        /// <summary>Saves the network as a C header with constants.</summary>
        /// <param name="path">Target <c>.h</c> file.</param>
        /// <param name="prefix">Prefix for all C symbols, e.g. "XOR".</param>
        public void DumpInitFile (string path, string prefix)
        {
            WriteText(path, NetworkSerializer.ToCHeader(ToData(), prefix));
        }

        private static void WriteText (string path, string text)
        {
            // UTF-8 without BOM and the serializer's LF line endings: the same bytes as the Python version
            File.WriteAllText(path, text, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        }
    }
}
