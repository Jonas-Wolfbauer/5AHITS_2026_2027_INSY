using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NeuralNetworkFromScratch
{
    /// <summary>
    /// Text formats for networks: JSON, human-readable TXT and C header.
    /// Writes byte-identical files to the Python version, so models can be exchanged and compared.
    /// </summary>
    public static partial class NetworkSerializer
    {
        /// <summary>Format name at the start of every file.</summary>
        public const string FormatName = "insy-nn";

        /// <summary>Format version; changes when the layout changes.</summary>
        public const int FormatVersion = 1;

        /// <summary>Checks that the parameters are complete and their shapes fit together.</summary>
        /// <exception cref="FormatException">A layer is empty or a neuron has the wrong number of weights.</exception>
        public static void Validate (NetworkData data)
        {
            if (data.InputSize < 1)
            {
                throw new FormatException($"invalid input size {data.InputSize}");
            }

            if (data.Layers.Count == 0)
            {
                throw new FormatException("network has no layers");
            }

            int expectedInputs = data.InputSize;
            for (int l = 0; l < data.Layers.Count; l++)
            {
                IReadOnlyList<NeuronData> neurons = data.Layers[l].Neurons;
                int layerNumber = l + 1;
                if (neurons.Count == 0)
                {
                    throw new FormatException($"layer {layerNumber} has no neurons");
                }

                foreach (NeuronData neuron in neurons)
                {
                    if (neuron.Weights.Length != expectedInputs)
                    {
                        throw new FormatException(
                            $"layer {layerNumber}: expected {expectedInputs} weights per neuron, got {neuron.Weights.Length}");
                    }
                }

                expectedInputs = neurons.Count;
            }
        }

        /// <summary>Serializes a network to indented JSON.</summary>
        public static string ToJson (NetworkData data)
        {
            // Written by hand to get exactly the layout of Python's json.dumps(data, indent=2):
            // one value per line, numbers as in FormatFloat
            var json = new StringBuilder();
            json.Append("{\n");
            json.Append($"  \"format\": \"{FormatName}\",\n");
            json.Append($"  \"version\": {FormatVersion},\n");
            json.Append($"  \"activation\": \"{data.Activation}\",\n");
            json.Append($"  \"input_size\": {data.InputSize},\n");
            json.Append("  \"layers\": [\n");

            for (int l = 0; l < data.Layers.Count; l++)
            {
                IReadOnlyList<NeuronData> neurons = data.Layers[l].Neurons;
                json.Append("    {\n");
                json.Append("      \"neurons\": [\n");

                for (int n = 0; n < neurons.Count; n++)
                {
                    double[] weights = neurons[n].Weights;
                    json.Append("        {\n");
                    json.Append("          \"weights\": [\n");
                    for (int w = 0; w < weights.Length; w++)
                    {
                        json.Append("            ").Append(FormatFloat(weights[w]));
                        json.Append(w < weights.Length - 1 ? ",\n" : "\n");
                    }

                    json.Append("          ],\n");
                    json.Append($"          \"bias\": {FormatFloat(neurons[n].Bias)}\n");
                    json.Append(n < neurons.Count - 1 ? "        },\n" : "        }\n");
                }

                json.Append("      ]\n");
                json.Append(l < data.Layers.Count - 1 ? "    },\n" : "    }\n");
            }

            json.Append("  ]\n");
            json.Append("}\n");
            return json.ToString();
        }

        /// <summary>Parses and validates JSON written by <see cref="ToJson"/> (in C# or Python).</summary>
        /// <exception cref="FormatException">The text is no valid network.</exception>
        public static NetworkData FromJson (string text)
        {
            try
            {
                using JsonDocument document = JsonDocument.Parse(text);
                JsonElement root = document.RootElement;

                if (root.ValueKind != JsonValueKind.Object
                    || !root.TryGetProperty("format", out JsonElement format)
                    || format.ValueKind != JsonValueKind.String
                    || format.GetString() != FormatName
                    || !root.TryGetProperty("version", out JsonElement version)
                    || version.ValueKind != JsonValueKind.Number
                    || version.GetInt32() != FormatVersion)
                {
                    throw new FormatException($"not an {FormatName} v{FormatVersion} network");
                }

                string activation = Require(root, "activation").GetString() ?? "";
                int inputSize = Require(root, "input_size").GetInt32();

                var layers = new List<LayerData>();
                foreach (JsonElement layer in Require(root, "layers").EnumerateArray())
                {
                    var neurons = new List<NeuronData>();
                    foreach (JsonElement neuron in Require(layer, "neurons").EnumerateArray())
                    {
                        double[] weights = Require(neuron, "weights").EnumerateArray()
                            .Select(weight => weight.GetDouble())
                            .ToArray();
                        double bias = Require(neuron, "bias").GetDouble();
                        neurons.Add(new NeuronData(weights, bias));
                    }

                    layers.Add(new LayerData(neurons));
                }

                var data = new NetworkData(activation, inputSize, layers);
                Validate(data);
                return data;
            }
            catch (Exception ex) when (ex is JsonException or InvalidOperationException)
            {
                // Syntax errors and values of the wrong type: one exception type for every broken file
                throw new FormatException($"invalid network JSON: {ex.Message}", ex);
            }
        }

        /// <summary>
        /// Formats a network as a human-readable, whitespace-separated text file. Every value is preceded
        /// by a keyword, so the file reads like a listing and a C program can parse it with <c>fscanf</c>.
        /// </summary>
        public static string ToTxt (NetworkData data)
        {
            var lines = new List<string>
            {
                $"{FormatName} {FormatVersion}",
                $"activation {data.Activation}",
                $"input_size {data.InputSize}",
                $"layers {data.Layers.Count}",
            };

            int inputCount = data.InputSize;
            for (int l = 0; l < data.Layers.Count; l++)
            {
                IReadOnlyList<NeuronData> neurons = data.Layers[l].Neurons;
                lines.Add("");
                lines.Add($"layer {l + 1} neurons {neurons.Count} inputs {inputCount}");
                for (int n = 0; n < neurons.Count; n++)
                {
                    string weights = string.Join(" ", neurons[n].Weights.Select(FormatFloat));
                    lines.Add($"neuron {n + 1} bias {FormatFloat(neurons[n].Bias)} weights {weights}");
                }

                inputCount = neurons.Count;
            }

            return string.Join("\n", lines) + "\n";
        }

        /// <summary>Parses a file written by <see cref="ToTxt"/> (in C# or Python).</summary>
        /// <exception cref="FormatException">The text is no valid network.</exception>
        public static NetworkData FromTxt (string text)
        {
            var tokens = new TokenReader(text);

            tokens.Expect(FormatName);
            if (tokens.NextInt() != FormatVersion)
            {
                throw new FormatException($"not an {FormatName} v{FormatVersion} network");
            }

            tokens.Expect("activation");
            string activation = tokens.Next();
            tokens.Expect("input_size");
            int inputSize = tokens.NextInt();
            tokens.Expect("layers");
            int layerCount = tokens.NextInt();

            var layers = new List<LayerData>();
            for (int layerNumber = 1; layerNumber <= layerCount; layerNumber++)
            {
                tokens.Expect("layer", layerNumber.ToString(CultureInfo.InvariantCulture), "neurons");
                int neuronCount = tokens.NextInt();
                tokens.Expect("inputs");
                int inputCount = tokens.NextInt();

                var neurons = new List<NeuronData>();
                for (int neuronNumber = 1; neuronNumber <= neuronCount; neuronNumber++)
                {
                    tokens.Expect("neuron", neuronNumber.ToString(CultureInfo.InvariantCulture), "bias");
                    double bias = tokens.NextDouble();
                    tokens.Expect("weights");
                    double[] weights = new double[inputCount];
                    for (int w = 0; w < inputCount; w++)
                    {
                        weights[w] = tokens.NextDouble();
                    }

                    neurons.Add(new NeuronData(weights, bias));
                }

                layers.Add(new LayerData(neurons));
            }

            var data = new NetworkData(activation, inputSize, layers);
            Validate(data);
            return data;
        }

        /// <summary>Generates a C header with all constants needed to rebuild the network.</summary>
        /// <param name="data">Parameters of the network.</param>
        /// <param name="prefix">
        /// Name prefix for every symbol, e.g. "XOR"; lets several networks live in one C program.
        /// </param>
        /// <exception cref="ArgumentException">The prefix is no valid C identifier.</exception>
        public static string ToCHeader (NetworkData data, string prefix)
        {
            prefix = prefix.ToUpperInvariant();
            if (!CIdentifier().IsMatch(prefix))
            {
                throw new ArgumentException($"prefix '{prefix}' is not a valid C identifier", nameof(prefix));
            }

            IReadOnlyList<LayerData> layers = data.Layers;
            string sizes = string.Join(", ", layers.Select(layer => layer.Neurons.Count));
            string guard = $"{prefix}_NET_H";

            var lines = new List<string>
            {
                $"// Generated by neural_network ({FormatName} v{FormatVersion}). Do not edit.",
                $"// Activation: {data.Activation}",
                $"#ifndef {guard}",
                $"#define {guard}",
                "",
                "#include \"neural_network.h\"",
                "",
                $"#define {prefix}_INPUT_SIZE {data.InputSize}",
                $"#define {prefix}_LAYER_COUNT {layers.Count}",
                "",
                $"static const int {prefix}_LAYER_SIZES[{prefix}_LAYER_COUNT] = {{{sizes}}};",
            };

            int inputCount = data.InputSize;
            for (int l = 0; l < layers.Count; l++)
            {
                IReadOnlyList<NeuronData> neurons = layers[l].Neurons;
                string symbol = $"{prefix}_L{l + 1}";
                lines.Add("");
                lines.Add($"// Layer {l + 1}: {neurons.Count} neurons x {inputCount} inputs");
                lines.Add($"static const double {symbol}_WEIGHTS[{neurons.Count}][{inputCount}] = {{");
                foreach (NeuronData neuron in neurons)
                {
                    lines.Add($"    {{{string.Join(", ", neuron.Weights.Select(FormatFloat))}}},");
                }

                lines.Add("};");
                string biases = string.Join(", ", neurons.Select(neuron => FormatFloat(neuron.Bias)));
                lines.Add($"static const double {symbol}_BIASES[{neurons.Count}] = {{{biases}}};");
                inputCount = neurons.Count;
            }

            lines.Add("");
            lines.AddRange(CInitFunction(data, prefix));
            lines.Add("");
            lines.Add($"#endif // {guard}");
            return string.Join("\n", lines) + "\n";
        }

        /// <summary>
        /// Formats a number exactly like Python's <c>repr</c>: the shortest text that parses back to the
        /// identical double, in Python's notation ("1.0", "0.0001", "1e-05", "1e+16").
        /// </summary>
        /// <exception cref="ArgumentException">The number is NaN or infinite; no format can store those.</exception>
        public static string FormatFloat (double value)
        {
            if (!double.IsFinite(value))
            {
                throw new ArgumentException($"cannot store {value}", nameof(value));
            }

            if (value == 0.0)
            {
                return double.IsNegative(value) ? "-0.0" : "0.0";
            }

            string sign = value < 0.0 ? "-" : "";
            (string digits, int pointPosition) = ShortestDigits(Math.Abs(value));

            // Python's rule: scientific notation below 1e-4 and from 1e16 on
            if (pointPosition <= -4 || pointPosition > 16)
            {
                string mantissa = digits.Length == 1 ? digits : $"{digits[0]}.{digits[1..]}";
                int exponent = pointPosition - 1;
                string exponentSign = exponent < 0 ? "-" : "+";
                return $"{sign}{mantissa}e{exponentSign}{Math.Abs(exponent):00}";
            }

            if (pointPosition <= 0)
            {
                return $"{sign}0.{new string('0', -pointPosition)}{digits}";
            }

            if (pointPosition >= digits.Length)
            {
                return $"{sign}{digits}{new string('0', pointPosition - digits.Length)}.0";
            }

            return $"{sign}{digits[..pointPosition]}.{digits[pointPosition..]}";
        }

        // Splits a positive number into its shortest round-trip digits and the position of the decimal
        // point relative to the first digit: 123.45 -> ("12345", 3), 0.001 -> ("1", -2).
        // "R" finds those digits, but switches to exponent notation at other thresholds than Python.
        private static (string Digits, int PointPosition) ShortestDigits (double value)
        {
            string text = value.ToString("R", CultureInfo.InvariantCulture);

            int exponent = 0;
            int exponentIndex = text.IndexOf('E');
            if (exponentIndex >= 0)
            {
                exponent = int.Parse(text[(exponentIndex + 1)..], CultureInfo.InvariantCulture);
                text = text[..exponentIndex];
            }

            int pointIndex = text.IndexOf('.');
            string digits = pointIndex < 0 ? text : text.Remove(pointIndex, 1);
            int pointPosition = (pointIndex < 0 ? text.Length : pointIndex) + exponent;

            // Leading zeros (0.001) shift the point, trailing zeros (100) carry no digits
            int leadingZeros = digits.Length - digits.TrimStart('0').Length;
            return (digits[leadingZeros..].TrimEnd('0'), pointPosition - leadingZeros);
        }

        // Builds the structs of c/neural_network.h around the constant arrays: no heap memory
        // and no file, as needed on a microcontroller
        private static List<string> CInitFunction (NetworkData data, string prefix)
        {
            IReadOnlyList<LayerData> layers = data.Layers;
            var lines = new List<string> { $"// Storage for the network that {prefix}_Init builds; no heap memory needed" };
            for (int l = 0; l < layers.Count; l++)
            {
                lines.Add($"static Neuron {prefix}_L{l + 1}_NEURONS[{layers[l].Neurons.Count}];");
            }

            lines.Add($"static Layer {prefix}_LAYERS[{prefix}_LAYER_COUNT];");
            lines.Add("");
            lines.Add("/// Builds the network from the constants above.");
            lines.Add("/// The weights stay in the constant arrays: never pass it to NetworkFree.");
            lines.Add($"static inline void {prefix}_Init (Network *network)");
            lines.Add("{");

            int inputCount = data.InputSize;
            for (int l = 0; l < layers.Count; l++)
            {
                int count = layers[l].Neurons.Count;
                string symbol = $"{prefix}_L{l + 1}";
                lines.Add($"    for (int n = 0; n < {count}; n++)");
                lines.Add("    {");
                lines.Add(
                    $"        {symbol}_NEURONS[n] = (Neuron){{.weights = {symbol}_WEIGHTS[n], "
                    + $".bias = {symbol}_BIASES[n], .inputCount = {inputCount}}};");
                lines.Add("    }");
                lines.Add(
                    $"    {prefix}_LAYERS[{l}] = (Layer){{.neurons = {symbol}_NEURONS, "
                    + $".neuronCount = {count}, .inputCount = {inputCount}}};");
                inputCount = count;
            }

            lines.Add(
                $"    *network = (Network){{.layers = {prefix}_LAYERS, "
                + $".layerCount = {prefix}_LAYER_COUNT, .inputSize = {prefix}_INPUT_SIZE}};");
            lines.Add("}");
            return lines;
        }

        private static JsonElement Require (JsonElement element, string name)
        {
            if (element.ValueKind != JsonValueKind.Object || !element.TryGetProperty(name, out JsonElement value))
            {
                throw new FormatException($"missing \"{name}\"");
            }

            return value;
        }

        [GeneratedRegex("^[A-Z_][A-Z0-9_]*$")]
        private static partial Regex CIdentifier ();

        /// <summary>Reads a text file token by token, like <c>fscanf</c> in C.</summary>
        private sealed class TokenReader
        {
            private readonly string[] _tokens;
            private int _position;

            public TokenReader (string text)
            {
                _tokens = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            }

            public string Next ()
            {
                if (_position >= _tokens.Length)
                {
                    throw new FormatException("unexpected end of file");
                }

                return _tokens[_position++];
            }

            public int NextInt ()
            {
                return int.Parse(Next(), CultureInfo.InvariantCulture);
            }

            public double NextDouble ()
            {
                return double.Parse(Next(), CultureInfo.InvariantCulture);
            }

            public void Expect (params string[] keywords)
            {
                foreach (string keyword in keywords)
                {
                    string token = Next();
                    if (token != keyword)
                    {
                        throw new FormatException($"expected '{keyword}', got '{token}'");
                    }
                }
            }
        }
    }
}
