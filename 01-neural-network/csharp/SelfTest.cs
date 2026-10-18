using System.Globalization;
using System.Text.RegularExpressions;

namespace NeuralNetworkFromScratch
{
    /// <summary>
    /// Built-in checks instead of a test framework (which would need NuGet packages): the same checks as
    /// the Python tests, plus loading the networks trained in Python.
    /// </summary>
    public static class SelfTest
    {
        private const double Epsilon = 1e-6;

        private static readonly double[][] _samples =
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0.3, -2.0, 7.5, 1.0],
        ];

        /// <summary>Runs all checks, prints one line per check and returns the process exit code.</summary>
        /// <param name="pythonModelDirectory">Folder with <c>xor</c> and <c>seven_segment</c> exported from Python.</param>
        public static int Run (string pythonModelDirectory)
        {
            (string Name, Action Check)[] checks =
            [
                ("sigmoid is stable for extreme inputs", CheckSigmoid),
                ("forward pass matches manual computation", CheckForwardPass),
                ("same seed gives same network", CheckSameSeed),
                ("invalid layer sizes are rejected", CheckInvalidLayerSizes),
                ("wrong input or target size is rejected", CheckWrongSizes),
                ("backpropagation matches numeric gradient, 2-2-1", () => CheckGradient([2, 2, 1], [1, 0], [1])),
                ("backpropagation matches numeric gradient, 4-16-7",
                    () => CheckGradient([4, 16, 7], [0, 1, 1, 0], [1, 0, 1, 1, 1, 1, 1])),
                ("backpropagation matches numeric gradient, 3-5-4-2",
                    () => CheckGradient([3, 5, 4, 2], [0.2, -0.7, 1.5], [0.1, 0.9])),
                ("XOR converges", CheckXor),
                ("7-segment decoder learns all digits", CheckSevenSegment),
                ("numbers are written like Python's repr", CheckFloatFormat),
                ("JSON round trip is exact", CheckJsonRoundTrip),
                ("TXT round trip is exact", CheckTxtRoundTrip),
                ("C header contains every parameter", CheckCHeader),
                ("broken files are rejected", CheckBrokenFiles),
                ("networks trained in Python predict correctly", () => CheckPythonModels(pythonModelDirectory)),
                ("files written from Python models are byte-identical",
                    () => CheckByteIdentical(pythonModelDirectory)),
            ];

            int failed = 0;
            foreach ((string name, Action check) in checks)
            {
                try
                {
                    check();
                    Console.WriteLine($"ok    {name}");
                }
                catch (Exception ex)
                {
                    // A failing check must not stop the others
                    failed++;
                    Console.WriteLine($"FAIL  {name}: {ex.Message}");
                }
            }

            Console.WriteLine();
            Console.WriteLine($"{checks.Length - failed}/{checks.Length} checks passed");
            return failed == 0 ? 0 : 1;
        }

        private static void CheckSigmoid ()
        {
            Check(Activations.Sigmoid(0.0) == 0.5, "sigmoid(0) != 0.5");
            Check(Activations.Sigmoid(1000.0) == 1.0, "sigmoid(1000) != 1");
            Check(Activations.Sigmoid(-1000.0) == 0.0, "sigmoid(-1000) != 0");
            Check(Activations.SigmoidDerivative(0.5) == 0.25, "derivative at a = 0.5 != 0.25");
        }

        private static void CheckForwardPass ()
        {
            var network = new NeuralNetwork([2, 2, 1], seed: 1);
            double[] inputs = [0.3, -0.8];

            double[] hidden = network.Layers[0].Neurons
                .Select(neuron => Activations.Sigmoid(neuron.Weights[0] * inputs[0] + neuron.Weights[1] * inputs[1] + neuron.Bias))
                .ToArray();
            Neuron output = network.Layers[1].Neurons[0];
            double expected = Activations.Sigmoid(output.Weights[0] * hidden[0] + output.Weights[1] * hidden[1] + output.Bias);

            Check(Math.Abs(network.Predict(inputs)[0] - expected) <= 1e-15, "prediction differs from the formula");
        }

        private static void CheckSameSeed ()
        {
            var first = new NeuralNetwork([4, 16, 7], seed: 3);
            var second = new NeuralNetwork([4, 16, 7], seed: 3);

            Check(first.Predict([1, 0, 0, 1]).SequenceEqual(second.Predict([1, 0, 0, 1])), "predictions differ");
        }

        private static void CheckInvalidLayerSizes ()
        {
            ExpectThrows<ArgumentException>(() => new NeuralNetwork([]), "no layer sizes");
            ExpectThrows<ArgumentException>(() => new NeuralNetwork([2]), "input size only");
            ExpectThrows<ArgumentException>(() => new NeuralNetwork([2, 0, 1]), "empty layer");
        }

        private static void CheckWrongSizes ()
        {
            var network = new NeuralNetwork([2, 2, 1], seed: 0);

            ExpectThrows<ArgumentException>(() => network.Predict([1, 0, 1]), "3 inputs for 2");
            ExpectThrows<ArgumentException>(() => network.Train([1, 0], [1, 0]), "2 targets for 1");
        }

        private static void CheckGradient (int[] layerSizes, double[] inputs, double[] targets)
        {
            var network = new NeuralNetwork(layerSizes, seed: 7);

            // With alpha = 1 an update is w_new = w_old - dJ/dw, so the difference is exactly the
            // gradient that backpropagation computed
            NeuralNetwork trained = NeuralNetwork.FromData(network.ToData(), learningRate: 1.0);
            trained.Train(inputs, targets);

            for (int l = 0; l < network.Layers.Length; l++)
            {
                for (int n = 0; n < network.Layers[l].NeuronCount; n++)
                {
                    Neuron neuron = network.Layers[l].Neurons[n];
                    Neuron updated = trained.Layers[l].Neurons[n];

                    for (int i = 0; i < neuron.Weights.Length; i++)
                    {
                        double original = neuron.Weights[i];
                        neuron.Weights[i] = original + Epsilon;
                        double costPlus = NeuralNetwork.Cost(network.Predict(inputs), targets);
                        neuron.Weights[i] = original - Epsilon;
                        double costMinus = NeuralNetwork.Cost(network.Predict(inputs), targets);
                        neuron.Weights[i] = original;

                        double numeric = (costPlus - costMinus) / (2 * Epsilon);
                        double analytic = original - updated.Weights[i];
                        Check(Math.Abs(numeric - analytic) <= 1e-8, $"layer {l + 1}, neuron {n + 1}, weight {i + 1}: {analytic} vs {numeric}");
                    }

                    double originalBias = neuron.Bias;
                    neuron.Bias = originalBias + Epsilon;
                    double biasCostPlus = NeuralNetwork.Cost(network.Predict(inputs), targets);
                    neuron.Bias = originalBias - Epsilon;
                    double biasCostMinus = NeuralNetwork.Cost(network.Predict(inputs), targets);
                    neuron.Bias = originalBias;

                    double numericBias = (biasCostPlus - biasCostMinus) / (2 * Epsilon);
                    double analyticBias = originalBias - updated.Bias;
                    Check(Math.Abs(numericBias - analyticBias) <= 1e-8, $"layer {l + 1}, neuron {n + 1}, bias: {analyticBias} vs {numericBias}");
                }
            }
        }

        private static void CheckXor ()
        {
            var network = new NeuralNetwork([2, 2, 1], learningRate: 0.1, seed: Program.XorSeed);

            List<double> losses = network.Fit(Datasets.XorInputs, Datasets.XorTargets, epochs: 20000);

            Check(losses[^1] < losses[0], "cost did not decrease");
            for (int s = 0; s < Datasets.XorInputs.Length; s++)
            {
                Check(Math.Round(network.Predict(Datasets.XorInputs[s])[0]) == Datasets.XorTargets[s][0], $"wrong output for sample {s}");
            }
        }

        private static void CheckSevenSegment ()
        {
            var network = new NeuralNetwork([4, 16, 7], learningRate: 0.1, seed: Program.SevenSegmentSeed);

            network.Fit(Datasets.SevenSegmentInputs, Datasets.SevenSegmentTargets, epochs: 20000);

            CheckDigits(network, "trained network");
        }

        private static void CheckFloatFormat ()
        {
            // Expected strings printed by Python's repr for the same doubles
            (double Value, string Python)[] cases =
            [
                (0.1, "0.1"), (-0.5, "-0.5"), (1.0, "1.0"), (123.0, "123.0"), (100.0, "100.0"),
                (0.0, "0.0"), (-0.0, "-0.0"), (1e-05, "1e-05"), (1.5e-05, "1.5e-05"), (0.0001, "0.0001"),
                (0.00012345, "0.00012345"), (1e15, "1000000000000000.0"), (1e16, "1e+16"),
                (1.2345678901234568e+16, "1.2345678901234568e+16"), (123456789.125, "123456789.125"),
                (5e-324, "5e-324"), (double.MaxValue, "1.7976931348623157e+308"),
                (-2.2250738585072014e-308, "-2.2250738585072014e-308"), (Math.Pow(2, -30), "9.313225746154785e-10"),
                (1.0 / 3.0, "0.3333333333333333"), (0.1 + 0.2, "0.30000000000000004"),
                (-4.0624992872092545, "-4.0624992872092545"),
            ];
            foreach ((double value, string python) in cases)
            {
                string text = NetworkSerializer.FormatFloat(value);
                Check(text == python, $"{python} written as {text}");
            }

            // Every written number must parse back to the identical double
            var random = new Random(5);
            for (int i = 0; i < 10000; i++)
            {
                double value = (random.NextDouble() - 0.5) * Math.Pow(10, random.Next(-30, 31));
                string text = NetworkSerializer.FormatFloat(value);
                Check(double.Parse(text, CultureInfo.InvariantCulture) == value, $"{text} does not parse back exactly");
            }
        }

        private static void CheckJsonRoundTrip ()
        {
            var network = new NeuralNetwork([4, 5, 3], seed: 11);

            string json = NetworkSerializer.ToJson(network.ToData());
            NeuralNetwork loaded = NeuralNetwork.FromData(NetworkSerializer.FromJson(json));

            Check(loaded.LayerSizes.SequenceEqual(network.LayerSizes), "layer sizes differ");
            CheckSamePredictions(network, loaded);
        }

        private static void CheckTxtRoundTrip ()
        {
            var network = new NeuralNetwork([4, 5, 3], seed: 11);

            string txt = NetworkSerializer.ToTxt(network.ToData());
            NeuralNetwork loaded = NeuralNetwork.FromData(NetworkSerializer.FromTxt(txt));

            string[] lines = txt.Split('\n');
            Check(lines[..4].SequenceEqual(["insy-nn 1", "activation sigmoid", "input_size 4", "layers 2"]), "unexpected header lines");
            Check(lines.Contains("layer 2 neurons 3 inputs 5"), "layer line missing");
            CheckSamePredictions(network, loaded);
        }

        private static void CheckCHeader ()
        {
            NetworkData data = new NeuralNetwork([4, 5, 3], seed: 11).ToData();

            string header = NetworkSerializer.ToCHeader(data, "demo");

            Check(header.Contains("#ifndef DEMO_NET_H"), "include guard missing");
            Check(header.Contains("#define DEMO_INPUT_SIZE 4"), "input size missing");
            Check(header.Contains("DEMO_LAYER_SIZES[DEMO_LAYER_COUNT] = {5, 3};"), "layer sizes missing");
            Check(header.Contains("DEMO_L1_WEIGHTS[5][4]") && header.Contains("DEMO_L2_BIASES[3]"), "arrays missing");
            Check(header.Contains("static inline void DEMO_Init (Network *network)"), "init function missing");

            // Every number in the arrays must parse back to the exact stored double, in file order
            int sizesIndex = header.IndexOf("DEMO_LAYER_SIZES", StringComparison.Ordinal);
            string body = header[(header.IndexOf("};", sizesIndex, StringComparison.Ordinal) + 2)..];
            double[] numbers = Regex.Matches(body, @"-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+e[-+]?\d+")
                .Select(match => double.Parse(match.Value, CultureInfo.InvariantCulture))
                .ToArray();

            var expected = new List<double>();
            foreach (LayerData layer in data.Layers)
            {
                foreach (NeuronData neuron in layer.Neurons)
                {
                    expected.AddRange(neuron.Weights);
                }

                expected.AddRange(layer.Neurons.Select(neuron => neuron.Bias));
            }

            Check(numbers.SequenceEqual(expected), "numbers in the header differ from the network");
            ExpectThrows<ArgumentException>(() => NetworkSerializer.ToCHeader(data, "7seg"), "prefix '7seg'");
        }

        private static void CheckBrokenFiles ()
        {
            NetworkData data = new NeuralNetwork([4, 5, 3], seed: 11).ToData();
            NeuronData neuron = data.Layers[1].Neurons[0];
            List<NeuronData> neurons = [neuron with { Weights = neuron.Weights[..^1] }, .. data.Layers[1].Neurons.Skip(1)];
            NetworkData missingWeight = data with { Layers = [data.Layers[0], new LayerData(neurons)] };

            ExpectThrows<FormatException>(() => NeuralNetwork.FromData(missingWeight), "a missing weight");
            ExpectThrows<FormatException>(() => NetworkSerializer.FromJson("{\"format\": \"other\", \"version\": 1}"), "another format");
            ExpectThrows<FormatException>(() => NetworkSerializer.FromJson("{\"format\": \"insy-nn\""), "truncated JSON");
            ExpectThrows<FormatException>(
                () => NetworkSerializer.FromTxt("insy-nn 1\nactivation sigmoid\ninput_size 4\nlayer 1"), "truncated TXT");
        }

        private static void CheckPythonModels (string directory)
        {
            foreach (string extension in new[] { ".json", ".txt" })
            {
                NeuralNetwork xor = LoadModel(Path.Combine(directory, "xor" + extension));
                for (int s = 0; s < Datasets.XorInputs.Length; s++)
                {
                    Check(
                        Math.Round(xor.Predict(Datasets.XorInputs[s])[0]) == Datasets.XorTargets[s][0],
                        $"xor{extension}: wrong output for sample {s}");
                }

                CheckDigits(LoadModel(Path.Combine(directory, "seven_segment" + extension)), "seven_segment" + extension);
            }

            CheckSamePredictions(
                LoadModel(Path.Combine(directory, "seven_segment.json")),
                LoadModel(Path.Combine(directory, "seven_segment.txt")));
        }

        private static void CheckByteIdentical (string directory)
        {
            foreach (string name in new[] { "xor", "seven_segment" })
            {
                string json = ReadText(Path.Combine(directory, name + ".json"));
                NetworkData data = NetworkSerializer.FromJson(json);

                CheckSameText(NetworkSerializer.ToJson(data), json, name + ".json");
                CheckSameText(NetworkSerializer.ToTxt(data), ReadText(Path.Combine(directory, name + ".txt")), name + ".txt");
                CheckSameText(NetworkSerializer.ToCHeader(data, name), ReadText(Path.Combine(directory, name + ".h")), name + ".h");
            }
        }

        private static NeuralNetwork LoadModel (string path)
        {
            return path.EndsWith(".txt", StringComparison.Ordinal) ? NeuralNetwork.LoadTxt(path) : NeuralNetwork.Load(path);
        }

        private static string ReadText (string path)
        {
            // Git may check text files out with CRLF on Windows; the formats themselves use LF
            return File.ReadAllText(path).Replace("\r\n", "\n");
        }

        private static void CheckDigits (NeuralNetwork network, string description)
        {
            for (int digit = 0; digit < Datasets.SevenSegmentInputs.Length; digit++)
            {
                double[] outputs = network.Predict(Datasets.SevenSegmentInputs[digit]);
                for (int segment = 0; segment < outputs.Length; segment++)
                {
                    Check(
                        Math.Round(outputs[segment]) == Datasets.SevenSegmentTargets[digit][segment],
                        $"{description}: digit {digit}, segment {"abcdefg"[segment]} wrong");
                }
            }
        }

        private static void CheckSamePredictions (NeuralNetwork first, NeuralNetwork second)
        {
            foreach (double[] sample in _samples.Where(sample => sample.Length == first.InputSize))
            {
                Check(first.Predict(sample).SequenceEqual(second.Predict(sample)), "predictions differ");
            }

            foreach (double[] sample in Datasets.SevenSegmentInputs.Where(sample => sample.Length == first.InputSize))
            {
                Check(first.Predict(sample).SequenceEqual(second.Predict(sample)), "predictions differ");
            }
        }

        private static void CheckSameText (string actual, string expected, string file)
        {
            if (actual == expected)
            {
                return;
            }

            string[] actualLines = actual.Split('\n');
            string[] expectedLines = expected.Split('\n');
            int line = 0;
            while (line < actualLines.Length && line < expectedLines.Length && actualLines[line] == expectedLines[line])
            {
                line++;
            }

            string Show (string[] lines) => line < lines.Length ? $"'{lines[line]}'" : "end of file";
            throw new InvalidOperationException($"{file}, line {line + 1}: expected {Show(expectedLines)}, got {Show(actualLines)}");
        }

        private static void Check (bool condition, string message)
        {
            if (!condition)
            {
                throw new InvalidOperationException(message);
            }
        }

        private static void ExpectThrows<TException> (Action action, string description)
            where TException : Exception
        {
            try
            {
                action();
            }
            catch (TException)
            {
                return;
            }

            throw new InvalidOperationException($"{description} was accepted");
        }
    }
}
