"""A single neuron with weights, bias and an activation function."""

# Python 3.12 (used by the TensorFlow comparison) needs this for "-> Neuron"
from __future__ import annotations

import numpy as np

from neural_network._typing import FloatArray
from neural_network.activation import ACTIVATIONS


class Neuron:
    """Single neuron computing a = activation(w . x + b).

    Stores its last inputs and output, because backpropagation needs both.

    Parameters
    ----------
    input_size : int
        Number of inputs, i.e. neurons in the previous layer.
    rng : np.random.Generator | None
        Source for the initial weights; pass a seeded generator for
        reproducible runs.
    activation : str
        Key into ``ACTIVATIONS``.
    """

    def __init__(
        self,
        input_size: int,
        rng: np.random.Generator | None = None,
        activation: str = "sigmoid",
    ) -> None:
        rng = rng if rng is not None else np.random.default_rng()

        # Symmetric range around 0, so neurons start out pulling in both directions
        self.weights: FloatArray = rng.uniform(-1.0, 1.0, size=input_size)
        self.bias: float = float(rng.uniform(-1.0, 1.0))

        self.activation = activation
        self._activation_fn, self._derivative_fn = ACTIVATIONS[activation]

        # Unknown until the first forward pass
        self.inputs: FloatArray = np.zeros(input_size)
        self.output: float = 0.0

    @classmethod
    def from_parameters(
        cls, weights: FloatArray, bias: float, activation: str = "sigmoid"
    ) -> Neuron:
        """Create a neuron with given parameters, e.g. when loading a file.

        Parameters
        ----------
        weights : FloatArray
            One weight per input.
        bias : float
            Bias term.
        activation : str
            Key into ``ACTIVATIONS``.

        Returns
        -------
        Neuron
            Neuron using exactly these parameters.
        """
        neuron = cls(len(weights), activation=activation)
        neuron.weights = np.array(weights, dtype=np.float64)
        neuron.bias = float(bias)
        return neuron

    def activate(self, inputs: FloatArray) -> float:
        """Run the forward pass of this neuron.

        Parameters
        ----------
        inputs : FloatArray
            Outputs of the previous layer (or the network input).

        Returns
        -------
        float
            a = activation(z) with z = w . x + b.
        """
        self.inputs = inputs
        z = float(np.dot(self.weights, inputs)) + self.bias
        self.output = self._activation_fn(z)
        return self.output

    def activate_derivative(self) -> float:
        """Return da/dz for the last forward pass.

        Returns
        -------
        float
            Derivative of the activation at the stored output.
        """
        return self._derivative_fn(self.output)

    def update_weights(self, delta: float, learning_rate: float) -> None:
        """Take one gradient-descent step on the weights.

        Parameters
        ----------
        delta : float
            dJ/dz of this neuron.
        learning_rate : float
            Step size alpha.
        """
        # dJ/dw_i = dJ/dz * dz/dw_i = delta * x_i
        self.weights -= learning_rate * delta * self.inputs

    def update_bias(self, delta: float, learning_rate: float) -> None:
        """Take one gradient-descent step on the bias.

        Parameters
        ----------
        delta : float
            dJ/dz of this neuron.
        learning_rate : float
            Step size alpha.
        """
        # dz/db = 1, so dJ/db = delta
        self.bias -= learning_rate * delta
