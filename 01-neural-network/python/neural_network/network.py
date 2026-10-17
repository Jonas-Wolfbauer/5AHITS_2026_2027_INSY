"""Feedforward neural network built from layers."""

from collections.abc import Callable, Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from neural_network import persistence
from neural_network._typing import FloatArray
from neural_network.layer import Layer
from neural_network.neuron import Neuron
from neural_network.persistence import NetworkDict


class NeuralNetwork:
    """Feedforward network trained with stochastic gradient descent.

    Parameters
    ----------
    layer_sizes : Sequence[int]
        Input size followed by the neuron count of every layer, e.g.
        ``[2, 2, 1]`` for 2 inputs, 2 hidden neurons and 1 output. The input
        is not a layer of its own: it has no weights.
    learning_rate : float
        Step size alpha for every update.
    seed : int | None
        Seed for the initial weights; ``None`` gives a different net per run.
    activation : str
        Activation of all neurons.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        learning_rate: float = 0.1,
        seed: int | None = None,
        activation: str = "sigmoid",
    ) -> None:
        if len(layer_sizes) < 2 or any(size < 1 for size in layer_sizes):
            raise ValueError(f"invalid layer sizes: {list(layer_sizes)}")

        self.learning_rate = learning_rate
        self.activation = activation

        rng = np.random.default_rng(seed)
        self.layers = [
            Layer(input_size, neuron_count, rng, activation)
            for input_size, neuron_count in pairwise(layer_sizes)
        ]

    @property
    def input_size(self) -> int:
        """Number of network inputs."""
        return self.layers[0].input_size

    @property
    def layer_sizes(self) -> list[int]:
        """Input size followed by the neuron count of every layer."""
        return [self.input_size, *(layer.neuron_count for layer in self.layers)]

    def predict(self, inputs: ArrayLike) -> FloatArray:
        """Run a forward pass.

        Parameters
        ----------
        inputs : ArrayLike
            One sample with ``input_size`` values.

        Returns
        -------
        FloatArray
            Activations of the output layer.
        """
        outputs = np.asarray(inputs, dtype=np.float64)
        if outputs.shape != (self.input_size,):
            raise ValueError(
                f"expected {self.input_size} inputs, got shape {outputs.shape}"
            )

        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def train(self, inputs: ArrayLike, targets: ArrayLike) -> float:
        """Run one SGD step: forward pass, backward pass, update.

        Parameters
        ----------
        inputs : ArrayLike
            One sample with ``input_size`` values.
        targets : ArrayLike
            Expected output for this sample.

        Returns
        -------
        float
            Cost J = 1/2 * sum((y_hat - y)^2) before the update.
        """
        expected = np.asarray(targets, dtype=np.float64)
        outputs = self.predict(inputs)
        if expected.shape != outputs.shape:
            raise ValueError(
                f"expected {outputs.shape[0]} targets, got shape {expected.shape}"
            )

        # dJ/dy_hat for J = 1/2 * (y_hat - y)^2
        errors = outputs - expected
        for layer in reversed(self.layers):
            errors = layer.backward(errors, self.learning_rate)

        return 0.5 * float(np.sum((outputs - expected) ** 2))

    def fit(
        self,
        inputs: Sequence[ArrayLike],
        targets: Sequence[ArrayLike],
        epochs: int,
        on_epoch: Callable[[int, float], None] | None = None,
    ) -> list[float]:
        """Train on a whole dataset for a number of epochs.

        Parameters
        ----------
        inputs : Sequence[ArrayLike]
            All samples.
        targets : Sequence[ArrayLike]
            Expected output per sample.
        epochs : int
            Passes over the dataset; each pass updates once per sample.
        on_epoch : Callable[[int, float], None] | None
            Called after every epoch with the number of finished epochs
            (starting at 1) and the mean cost of that epoch, e.g. for
            progress output or recording snapshots.

        Returns
        -------
        list[float]
            Mean cost per epoch.
        """
        if len(inputs) != len(targets) or len(inputs) == 0:
            raise ValueError("inputs and targets must be non-empty and equally long")

        losses = []
        for epoch in range(1, epochs + 1):
            epoch_loss = sum(
                self.train(x, y) for x, y in zip(inputs, targets, strict=True)
            )
            losses.append(epoch_loss / len(inputs))
            if on_epoch is not None:
                on_epoch(epoch, losses[-1])
        return losses

    def mean_cost(
        self, inputs: Sequence[ArrayLike], targets: Sequence[ArrayLike]
    ) -> float:
        """Compute the mean cost over a dataset without training.

        Parameters
        ----------
        inputs : Sequence[ArrayLike]
            All samples.
        targets : Sequence[ArrayLike]
            Expected output per sample.

        Returns
        -------
        float
            Mean of J = 1/2 * sum((y_hat - y)^2) over all samples.
        """
        total = 0.0
        for x, y in zip(inputs, targets, strict=True):
            difference = self.predict(x) - np.asarray(y, dtype=np.float64)
            total += 0.5 * float(np.sum(difference**2))
        return total / len(inputs)

    def to_dict(self) -> NetworkDict:
        """Return all parameters in the shared file layout.

        Returns
        -------
        NetworkDict
            Plain dict, see ``neural_network.persistence``.
        """
        return {
            "format": persistence.FORMAT_NAME,
            "version": persistence.FORMAT_VERSION,
            "activation": self.activation,
            "input_size": self.input_size,
            "layers": [
                {
                    "neurons": [
                        {"weights": neuron.weights.tolist(), "bias": neuron.bias}
                        for neuron in layer.neurons
                    ]
                }
                for layer in self.layers
            ],
        }

    @classmethod
    def from_dict(cls, data: NetworkDict, learning_rate: float = 0.1) -> NeuralNetwork:
        """Rebuild a network from the shared file layout.

        Parameters
        ----------
        data : NetworkDict
            Plain dict, see ``neural_network.persistence``.
        learning_rate : float
            Step size for further training; not part of the file.

        Returns
        -------
        NeuralNetwork
            Network with exactly the stored parameters.
        """
        persistence.validate(data)
        layer_sizes = [
            data["input_size"],
            *(len(layer_data["neurons"]) for layer_data in data["layers"]),
        ]
        network = cls(layer_sizes, learning_rate, activation=data["activation"])

        for layer, layer_data in zip(network.layers, data["layers"], strict=True):
            layer.neurons = [
                Neuron.from_parameters(n["weights"], n["bias"], data["activation"])
                for n in layer_data["neurons"]
            ]
        return network

    def dump_json(self, path: str | Path) -> None:
        """Save the network as JSON.

        Parameters
        ----------
        path : str | Path
            Target file.
        """
        _write_text(path, persistence.to_json(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> NeuralNetwork:
        """Load a network saved with ``dump_json``.

        Parameters
        ----------
        path : str | Path
            JSON file.

        Returns
        -------
        NeuralNetwork
            The stored network.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(persistence.from_json(text))

    def dump_txt(self, path: str | Path) -> None:
        """Save the network as human-readable text.

        Parameters
        ----------
        path : str | Path
            Target file.
        """
        _write_text(path, persistence.to_txt(self.to_dict()))

    @classmethod
    def load_txt(cls, path: str | Path) -> NeuralNetwork:
        """Load a network saved with ``dump_txt``.

        Parameters
        ----------
        path : str | Path
            Text file.

        Returns
        -------
        NeuralNetwork
            The stored network.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(persistence.from_txt(text))

    def dump_initfile(self, path: str | Path, prefix: str) -> None:
        """Save the network as a C header with constants.

        Parameters
        ----------
        path : str | Path
            Target ``.h`` file.
        prefix : str
            Prefix for all C symbols, e.g. ``"XOR"``.
        """
        _write_text(path, persistence.to_c_header(self.to_dict(), prefix))


def _write_text(path: str | Path, text: str) -> None:
    # Fixed LF line endings, so files are identical on every OS
    Path(path).write_text(text, encoding="utf-8", newline="\n")
