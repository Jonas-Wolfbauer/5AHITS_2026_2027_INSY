namespace NeuralNetworkFromScratch
{
    /// <summary>Training data from the assignment (XOR gate, 7-segment decoder).</summary>
    public static class Datasets
    {
        /// <summary>All four input combinations of the XOR gate.</summary>
        public static readonly double[][] XorInputs = [[0, 0], [0, 1], [1, 0], [1, 1]];

        /// <summary>XOR of each input pair.</summary>
        public static readonly double[][] XorTargets = [[0], [1], [1], [0]];

        /// <summary>Digits 0–9 as 4-bit BCD.</summary>
        public static readonly double[][] SevenSegmentInputs =
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
        ];

        /// <summary>Segments a–g that light up for each digit.</summary>
        public static readonly double[][] SevenSegmentTargets =
        [
            [1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, 1],
            [1, 1, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1],
        ];
    }
}
