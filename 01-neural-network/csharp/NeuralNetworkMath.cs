namespace NeuralNetworkFromScratch
{
    /// <summary>Arithmetic of a neuron, written as a plain loop so every step can be followed in the debugger.</summary>
    public static class NeuralNetworkMath
    {
        /// <summary>Returns the weighted sum z = w · x + b of one neuron.</summary>
        /// <remarks>
        /// A SIMD version with <c>System.Numerics.Vector</c> would be faster, but hides the arithmetic;
        /// the networks of this assignment have at most 16 inputs per neuron.
        /// </remarks>
        public static double ComputeNeuron (double[] inputs, double[] weights, double bias)
        {
            double sum = 0.0;
            for (int i = 0; i < weights.Length; i++)
            {
                sum += weights[i] * inputs[i];
            }

            return sum + bias;
        }
    }
}
