namespace NeuralNetworkFromScratch
{
    /// <summary>Activation functions and their derivatives, looked up by the name stored in model files.</summary>
    public static class Activations
    {
        /// <summary>Name of the logistic sigmoid in model files.</summary>
        public const string SigmoidName = "sigmoid";

        /// <summary>Returns the activation function called <paramref name="name"/> and its derivative.</summary>
        /// <exception cref="ArgumentException">The name is unknown.</exception>
        public static (Func<double, double> Function, Func<double, double> Derivative) Get (string name)
        {
            return name switch
            {
                SigmoidName => (Sigmoid, SigmoidDerivative),
                _ => throw new ArgumentException($"unknown activation '{name}'", nameof(name)),
            };
        }

        /// <summary>Logistic sigmoid σ(z) = 1 / (1 + e^−z); maps any z into the open interval (0, 1).</summary>
        public static double Sigmoid (double z)
        {
            // Same two branches as the Python version: Exp only ever gets a non-positive argument
            if (z >= 0.0)
            {
                return 1.0 / (1.0 + Math.Exp(-z));
            }

            double expZ = Math.Exp(z);
            return expZ / (1.0 + expZ);
        }

        /// <summary>
        /// Derivative da/dz = a · (1 − a) of the sigmoid, computed from its own output
        /// <paramref name="output"/> = σ(z) — not from z, so σ must not be applied again.
        /// </summary>
        public static double SigmoidDerivative (double output)
        {
            return output * (1.0 - output);
        }
    }
}
