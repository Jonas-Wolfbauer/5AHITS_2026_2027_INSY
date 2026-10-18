"""Train one problem three ways and collect what the assignment asks to compare.

1. The own implementation (``neural_network`` from ``../python``).
2. TensorFlow starting from exactly the same weights. With the same cost, the
   same sample order and float64, both must follow the same path; any real
   difference would point to a bug in the own backpropagation.
3. TensorFlow with Keras' default initialization (Glorot uniform weights,
   zero biases), i.e. how TensorFlow is typically used.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import keras
import numpy as np
import tensorflow as tf

from neural_network import NeuralNetwork
from neural_network.datasets import (
    SEVEN_SEGMENT_INPUTS,
    SEVEN_SEGMENT_TARGETS,
    XOR_INPUTS,
    XOR_TARGETS,
)

LEARNING_RATE = 0.1
EPOCHS = 20000
# A run counts as converged once the mean cost drops below this
CONVERGED_COST = 0.01


@dataclass(frozen=True)
class Problem:
    """Network layout and data of one assignment problem."""

    name: str
    layer_sizes: list[int]
    seed: int
    inputs: list[list[int]]
    targets: list[list[int]]


@dataclass(frozen=True)
class Run:
    """Outcome of one training run."""

    label: str
    costs: np.ndarray
    outputs: np.ndarray
    seconds: float

    def correct(self, targets: Sequence[Sequence[int]]) -> int:
        """Count samples whose outputs all round to their targets.

        Parameters
        ----------
        targets : Sequence[Sequence[int]]
            Expected output per sample.

        Returns
        -------
        int
            Number of correctly predicted samples.
        """
        rounded = np.round(self.outputs) == np.asarray(targets)
        return int(np.sum(np.all(rounded, axis=1)))

    def converged_at(self) -> int | None:
        """Return the first epoch whose mean cost is below ``CONVERGED_COST``.

        Returns
        -------
        int | None
            Epoch number, or ``None`` if the run never got there.
        """
        below = np.flatnonzero(self.costs < CONVERGED_COST)
        return int(below[0]) if below.size else None


@dataclass(frozen=True)
class Comparison:
    """All runs of one problem."""

    problem: Problem
    epochs: int
    own: Run
    same_start: Run
    keras_default: Run

    @property
    def max_difference(self) -> float:
        """Largest output difference between own and same-start TensorFlow."""
        return float(np.max(np.abs(self.own.outputs - self.same_start.outputs)))


PROBLEMS = {
    "xor": Problem("XOR", [2, 2, 1], 0, XOR_INPUTS, XOR_TARGETS),
    "seven-segment": Problem(
        "7-segment", [4, 16, 7], 42, SEVEN_SEGMENT_INPUTS, SEVEN_SEGMENT_TARGETS
    ),
}


def compare(problem: Problem, epochs: int = EPOCHS) -> Comparison:
    """Train a problem with the own implementation and twice with TensorFlow.

    Parameters
    ----------
    problem : Problem
        Network layout, seed and data.
    epochs : int
        Passes over the data for every run.

    Returns
    -------
    Comparison
        Costs, outputs and run times of all three runs.
    """
    network = NeuralNetwork(problem.layer_sizes, LEARNING_RATE, problem.seed)

    # Copied before the own run changes the weights
    same_start_model = build_model(problem.layer_sizes)
    copy_weights(network, same_start_model)

    keras.utils.set_random_seed(problem.seed)
    default_model = build_model(problem.layer_sizes)

    return Comparison(
        problem=problem,
        epochs=epochs,
        own=_run_own(network, problem, epochs),
        same_start=_run_tensorflow(
            "TensorFlow, same start", same_start_model, problem, epochs
        ),
        keras_default=_run_tensorflow(
            f"TensorFlow, Keras init, seed {problem.seed}",
            default_model,
            problem,
            epochs,
        ),
    )


def build_model(layer_sizes: Sequence[int]) -> keras.Sequential:
    """Build a Keras model with the same layout as the own network.

    Parameters
    ----------
    layer_sizes : Sequence[int]
        Input size followed by the neuron count of every layer.

    Returns
    -------
    keras.Sequential
        Fully connected sigmoid layers in float64.
    """
    layers = [keras.Input(shape=(layer_sizes[0],), dtype="float64")]
    layers += [
        keras.layers.Dense(size, activation="sigmoid", dtype="float64")
        for size in layer_sizes[1:]
    ]
    return keras.Sequential(layers)


def copy_weights(network: NeuralNetwork, model: keras.Sequential) -> None:
    """Give the Keras model exactly the weights of the own network.

    Parameters
    ----------
    network : NeuralNetwork
        Source of the weights.
    model : keras.Sequential
        Model built by ``build_model`` with the same layer sizes.
    """
    for layer, dense in zip(network.layers, model.layers, strict=True):
        # Keras stores one column per neuron, the own network one row
        kernel = np.array([neuron.weights for neuron in layer.neurons]).T
        biases = np.array([neuron.bias for neuron in layer.neurons])
        dense.set_weights([kernel, biases])


def train_tensorflow(
    model: keras.Sequential,
    inputs: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    epochs: int,
    learning_rate: float = LEARNING_RATE,
) -> np.ndarray:
    """Train with plain SGD, one update per sample, in a fixed sample order.

    Parameters
    ----------
    model : keras.Sequential
        Model to train in place.
    inputs : Sequence[Sequence[float]]
        All samples.
    targets : Sequence[Sequence[float]]
        Expected output per sample.
    epochs : int
        Passes over the data.
    learning_rate : float
        Step size alpha.

    Returns
    -------
    np.ndarray
        Mean cost per epoch, as returned by ``NeuralNetwork.fit``.
    """
    x = tf.constant(inputs, dtype=tf.float64)
    y = tf.constant(targets, dtype=tf.float64)
    sample_count = len(inputs)
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer.build(model.trainable_variables)

    # One graph for the whole training: calling back into Python per sample
    # would cost more than the arithmetic of these tiny networks
    @tf.function
    def train(epoch_count: tf.Tensor) -> tf.Tensor:
        costs = tf.TensorArray(tf.float64, size=epoch_count)
        for epoch in tf.range(epoch_count):
            epoch_cost = tf.constant(0.0, dtype=tf.float64)
            for i in tf.range(sample_count):
                with tf.GradientTape() as tape:
                    output = model(x[i : i + 1], training=True)
                    # Same cost as the own implementation: J = 1/2 * sum((y_hat - y)^2)
                    cost = 0.5 * tf.reduce_sum(tf.square(output - y[i : i + 1]))
                gradients = tape.gradient(cost, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables, strict=True)
                )
                epoch_cost += cost
            costs = costs.write(epoch, epoch_cost / sample_count)
        return costs.stack()

    return train(tf.constant(epochs)).numpy()


def mean_cost(outputs: np.ndarray, targets: Sequence[Sequence[float]]) -> float:
    """Compute the mean of J = 1/2 * sum((y_hat - y)^2) over all samples.

    Parameters
    ----------
    outputs : np.ndarray
        Predictions, one row per sample.
    targets : Sequence[Sequence[float]]
        Expected output per sample.

    Returns
    -------
    float
        Mean cost.
    """
    difference = outputs - np.asarray(targets, dtype=np.float64)
    return float(np.mean(0.5 * np.sum(difference**2, axis=1)))


def _run_own(network: NeuralNetwork, problem: Problem, epochs: int) -> Run:
    start = time.perf_counter()
    initial = network.mean_cost(problem.inputs, problem.targets)
    costs = network.fit(problem.inputs, problem.targets, epochs)
    seconds = time.perf_counter() - start

    outputs = np.array([network.predict(sample) for sample in problem.inputs])
    label = f"Own implementation, seed {problem.seed}"
    return Run(label, np.array([initial, *costs]), outputs, seconds)


def _run_tensorflow(
    label: str, model: keras.Sequential, problem: Problem, epochs: int
) -> Run:
    # Includes building the graph (about a second), as a user would experience it
    start = time.perf_counter()
    initial = mean_cost(_predict(model, problem.inputs), problem.targets)
    costs = train_tensorflow(model, problem.inputs, problem.targets, epochs)
    seconds = time.perf_counter() - start

    outputs = _predict(model, problem.inputs)
    return Run(label, np.array([initial, *costs]), outputs, seconds)


def _predict(model: keras.Sequential, inputs: Sequence[Sequence[float]]) -> np.ndarray:
    return model(tf.constant(inputs, dtype=tf.float64), training=False).numpy()
