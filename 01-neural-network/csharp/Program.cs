using System.Diagnostics;
using System.Globalization;

namespace NeuralNetworkFromScratch
{
    /// <summary>Command line: train the assignment networks, export and load models, run the self-test.</summary>
    internal static class Program
    {
        /// <summary>Default seed for the XOR network; converges with <see cref="Random"/>.</summary>
        public const int XorSeed = 0;

        /// <summary>Default seed for the 7-segment network.</summary>
        public const int SevenSegmentSeed = 42;

        private const int DefaultEpochs = 20000;
        private const double DefaultLearningRate = 0.1;
        private const string DefaultPythonModels = "../models/python";
        private const string SegmentNames = "abcdefg";

        private const string Usage = """
            Neural network from scratch (INSY), C# version.

            usage: dotnet run -- <command> [options]

            commands:
              xor            train the XOR gate (2-2-1)
              seven-segment  train the 7-segment decoder (4-16-7)
              load FILE      load a .json or .txt model and run it on its dataset
              selftest       run the built-in checks

            options of xor and seven-segment:
              --epochs N     epochs (default 20000)
              --lr ALPHA     learning rate alpha (default 0.1)
              --seed S       seed for the initial weights (default 0 / 42)
              --hidden H     hidden neurons (default 2 / 16)
              --export DIR   write .json, .txt and .h to DIR and verify the reload

            options of selftest:
              --models DIR   networks exported from Python (default ../models/python)
            """;

        private static int Main (string[] args)
        {
            if (args.Length == 0 || args[0] is "-h" or "--help")
            {
                Console.WriteLine(Usage);
                return args.Length == 0 ? 2 : 0;
            }

            try
            {
                string[] options = args[1..];
                return args[0] switch
                {
                    "xor" => RunXor(ParseTrainingOptions(options, XorSeed, defaultHidden: 2)),
                    "seven-segment" => RunSevenSegment(ParseTrainingOptions(options, SevenSegmentSeed, defaultHidden: 16)),
                    "load" => RunLoad(options),
                    "selftest" => SelfTest.Run(ParseModelDirectory(options)),
                    _ => throw new ArgumentException($"unknown command '{args[0]}', see --help"),
                };
            }
            catch (Exception ex) when (ex is ArgumentException or FormatException or IOException or UnauthorizedAccessException)
            {
                // Wrong usage or unreadable files: a short message helps more than a stack trace
                Console.Error.WriteLine($"error: {ex.Message}");
                return 2;
            }
        }

        private static int RunXor (TrainingOptions options)
        {
            NeuralNetwork network = Train("XOR", [2, options.Hidden, 1], Datasets.XorInputs, Datasets.XorTargets, options);

            Console.WriteLine();
            bool allCorrect = PrintXorResults(network);

            ExportIfRequested(network, options, "xor", Datasets.XorInputs);
            return allCorrect ? 0 : 1;
        }

        private static int RunSevenSegment (TrainingOptions options)
        {
            NeuralNetwork network = Train(
                "7-segment", [4, options.Hidden, 7], Datasets.SevenSegmentInputs, Datasets.SevenSegmentTargets, options);

            Console.WriteLine();
            bool allCorrect = PrintSevenSegmentResults(network);

            ExportIfRequested(network, options, "seven_segment", Datasets.SevenSegmentInputs);
            return allCorrect ? 0 : 1;
        }

        private static int RunLoad (string[] args)
        {
            if (args.Length != 1)
            {
                throw new ArgumentException("usage: load FILE (.json or .txt)");
            }

            string path = args[0];
            NeuralNetwork network = path.EndsWith(".txt", StringComparison.OrdinalIgnoreCase)
                ? NeuralNetwork.LoadTxt(path)
                : NeuralNetwork.Load(path);
            Console.WriteLine($"loaded {path}: {Architecture(network)}");
            Console.WriteLine();

            int[] sizes = network.LayerSizes;
            return (sizes[0], sizes[^1]) switch
            {
                (2, 1) => PrintXorResults(network) ? 0 : 1,
                (4, 7) => PrintSevenSegmentResults(network) ? 0 : 1,
                _ => throw new ArgumentException($"no dataset for a {Architecture(network)} network"),
            };
        }

        private static NeuralNetwork Train (
            string name, int[] layerSizes, double[][] inputs, double[][] targets, TrainingOptions options)
        {
            var network = new NeuralNetwork(layerSizes, options.LearningRate, options.Seed);
            Console.WriteLine(
                $"{name} {Architecture(network)}, {options.Epochs} epochs, alpha {options.LearningRate}, seed {options.Seed}");
            Console.WriteLine($"epoch {0,6}  cost {network.MeanCost(inputs, targets):F6}");

            int reportEvery = Math.Max(1, options.Epochs / 10);
            var stopwatch = Stopwatch.StartNew();
            network.Fit(inputs, targets, options.Epochs, (epoch, cost) =>
            {
                if (epoch % reportEvery == 0 || epoch == options.Epochs)
                {
                    Console.WriteLine($"epoch {epoch,6}  cost {cost:F6}");
                }
            });
            Console.WriteLine($"trained in {stopwatch.Elapsed.TotalSeconds:F2} s");

            return network;
        }

        // Prints one row per XOR input; returns whether every output rounds to its target
        private static bool PrintXorResults (NeuralNetwork network)
        {
            Console.WriteLine("input    target  output   rounded");
            bool allCorrect = true;
            for (int s = 0; s < Datasets.XorInputs.Length; s++)
            {
                double output = network.Predict(Datasets.XorInputs[s])[0];
                double target = Datasets.XorTargets[s][0];
                bool correct = Math.Round(output) == target;
                allCorrect &= correct;
                Console.WriteLine(
                    $"{FormatSample(Datasets.XorInputs[s]),-8} {target,-7} {output:F4}   {Math.Round(output)}  {(correct ? "ok" : "WRONG")}");
            }

            return allCorrect;
        }

        // Draws all ten digits and lists the outputs; returns whether every digit is decoded correctly
        private static bool PrintSevenSegmentResults (NeuralNetwork network)
        {
            double[][] outputs = Datasets.SevenSegmentInputs.Select(network.Predict).ToArray();
            PrintDigits(outputs);
            Console.WriteLine();

            int correctDigits = 0;
            for (int digit = 0; digit < outputs.Length; digit++)
            {
                var wrongSegments = new List<char>();
                for (int segment = 0; segment < outputs[digit].Length; segment++)
                {
                    if (Math.Round(outputs[digit][segment]) != Datasets.SevenSegmentTargets[digit][segment])
                    {
                        wrongSegments.Add(SegmentNames[segment]);
                    }
                }

                if (wrongSegments.Count == 0)
                {
                    correctDigits++;
                }

                string values = string.Join(" ", outputs[digit].Select(value => value.ToString("F2", CultureInfo.InvariantCulture)));
                string status = wrongSegments.Count == 0 ? "ok" : $"WRONG segments {string.Join(",", wrongSegments)}";
                Console.WriteLine($"{digit}: {values}  {status}");
            }

            Console.WriteLine();
            Console.WriteLine($"{correctDigits}/10 digits correct");
            return correctDigits == outputs.Length;
        }

        private static void PrintDigits (double[][] outputs)
        {
            string[] rows = ["", "", ""];
            foreach (double[] output in outputs)
            {
                bool[] on = output.Select(value => Math.Round(value) == 1.0).ToArray();
                rows[0] += " " + (on[0] ? "_" : " ") + "   ";
                rows[1] += (on[5] ? "|" : " ") + (on[6] ? "_" : " ") + (on[1] ? "|" : " ") + "  ";
                rows[2] += (on[4] ? "|" : " ") + (on[3] ? "_" : " ") + (on[2] ? "|" : " ") + "  ";
            }

            foreach (string row in rows)
            {
                Console.WriteLine(row.TrimEnd());
            }

            Console.WriteLine(string.Concat(Enumerable.Range(0, outputs.Length).Select(digit => $" {digit}   ")).TrimEnd());
        }

        private static void ExportIfRequested (NeuralNetwork network, TrainingOptions options, string name, double[][] inputs)
        {
            if (options.ExportDirectory is null)
            {
                return;
            }

            Directory.CreateDirectory(options.ExportDirectory);
            string jsonPath = Path.Combine(options.ExportDirectory, $"{name}.json");
            string txtPath = Path.Combine(options.ExportDirectory, $"{name}.txt");
            string headerPath = Path.Combine(options.ExportDirectory, $"{name}.h");

            network.DumpJson(jsonPath);
            network.DumpTxt(txtPath);
            network.DumpInitFile(headerPath, name);

            Console.WriteLine();
            (string Path, NeuralNetwork Loaded)[] reloaded =
            [
                (jsonPath, NeuralNetwork.Load(jsonPath)),
                (txtPath, NeuralNetwork.LoadTxt(txtPath)),
            ];
            foreach ((string path, NeuralNetwork loaded) in reloaded)
            {
                double difference = inputs.Max(sample => MaxDifference(network.Predict(sample), loaded.Predict(sample)));
                Console.WriteLine($"reloaded {path}: max difference {difference.ToString("0.0e+00", CultureInfo.InvariantCulture)}");
            }

            Console.WriteLine($"written  {headerPath}");
        }

        private static double MaxDifference (double[] first, double[] second)
        {
            return first.Zip(second, (a, b) => Math.Abs(a - b)).Max();
        }

        private static string Architecture (NeuralNetwork network)
        {
            return string.Join("-", network.LayerSizes);
        }

        private static string FormatSample (double[] sample)
        {
            return $"[{string.Join(", ", sample)}]";
        }

        private static TrainingOptions ParseTrainingOptions (string[] args, int defaultSeed, int defaultHidden)
        {
            var options = new TrainingOptions(DefaultEpochs, DefaultLearningRate, defaultSeed, defaultHidden, null);
            for (int i = 0; i < args.Length; i += 2)
            {
                string name = args[i];
                if (name is not ("--epochs" or "--lr" or "--seed" or "--hidden" or "--export"))
                {
                    throw new ArgumentException($"unknown option '{name}', see --help");
                }

                if (i + 1 >= args.Length)
                {
                    throw new ArgumentException($"option {name} needs a value");
                }

                string value = args[i + 1];
                options = name switch
                {
                    "--epochs" => options with { Epochs = int.Parse(value, CultureInfo.InvariantCulture) },
                    "--lr" => options with { LearningRate = double.Parse(value, CultureInfo.InvariantCulture) },
                    "--seed" => options with { Seed = int.Parse(value, CultureInfo.InvariantCulture) },
                    "--hidden" => options with { Hidden = int.Parse(value, CultureInfo.InvariantCulture) },
                    _ => options with { ExportDirectory = value },
                };
            }

            return options;
        }

        private static string ParseModelDirectory (string[] args)
        {
            return args switch
            {
                [] => DefaultPythonModels,
                ["--models", string directory] => directory,
                _ => throw new ArgumentException("usage: selftest [--models DIR]"),
            };
        }

        // Settings of the training commands, filled from the command line
        private sealed record TrainingOptions (int Epochs, double LearningRate, int Seed, int Hidden, string? ExportDirectory);
    }
}
