"""A fully connected layer of neurons."""

import numpy as np

from neural_network._typing import FloatArray
from neural_network.neuron import Neuron


class Layer:
    """Fully connected layer: every neuron sees every input.

    Parameters
    ----------
    input_size : int
        Number of inputs, i.e. neurons in the previous layer.
    neuron_count : int
        Number of neurons in this layer.
    rng : np.random.Generator | None
        Source for the initial weights.
    activation : str
        Activation of all neurons in this layer.
    """

    def __init__(
        self,
        input_size: int,
        neuron_count: int,
        rng: np.random.Generator | None = None,
        activation: str = "sigmoid",
    ) -> None:
        self.input_size = input_size
        self.neurons = [
            Neuron(input_size, rng, activation) for _ in range(neuron_count)
        ]

    @property
    def neuron_count(self) -> int:
        """Number of neurons in this layer."""
        return len(self.neurons)

    def forward(self, inputs: FloatArray) -> FloatArray:
        """Compute the output of every neuron.

        Parameters
        ----------
        inputs : FloatArray
            Outputs of the previous layer.

        Returns
        -------
        FloatArray
            One activation per neuron.
        """
        return np.array([neuron.activate(inputs) for neuron in self.neurons])

    def backward(self, errors: FloatArray, learning_rate: float) -> FloatArray:
        """Backpropagate through this layer and update its neurons.

        Parameters
        ----------
        errors : FloatArray
            dJ/da for every neuron of this layer.
        learning_rate : float
            Step size alpha.

        Returns
        -------
        FloatArray
            dJ/da for every neuron of the previous layer.
        """
        previous_errors = np.zeros(self.input_size)

        for neuron, error in zip(self.neurons, errors, strict=True):
            # dJ/dz = dJ/da * da/dz
            delta = error * neuron.activate_derivative()

            # dz/dx_i = w_i; must use the weights from the forward pass,
            # so this happens before the update below
            previous_errors += delta * neuron.weights

            neuron.update_weights(delta, learning_rate)
            neuron.update_bias(delta, learning_rate)

        return previous_errors
