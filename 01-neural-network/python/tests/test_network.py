"""Forward pass, input validation and convergence on the assignment datasets."""

import numpy as np
import pytest

from neural_network import NeuralNetwork
from neural_network.activation import sigmoid, sigmoid_derivative
from neural_network.datasets import (
    SEVEN_SEGMENT_INPUTS,
    SEVEN_SEGMENT_TARGETS,
    XOR_INPUTS,
    XOR_TARGETS,
)


def test_sigmoid_is_stable_for_extreme_inputs():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid_derivative(0.5) == 0.25


def test_forward_pass_matches_manual_computation():
    network = NeuralNetwork([2, 2, 1], seed=1)
    inputs = np.array([0.3, -0.8])

    hidden = np.array(
        [sigmoid(n.weights @ inputs + n.bias) for n in network.layers[0].neurons]
    )
    output_neuron = network.layers[1].neurons[0]
    expected = sigmoid(output_neuron.weights @ hidden + output_neuron.bias)

    assert network.predict(inputs)[0] == pytest.approx(expected, abs=1e-15)


def test_same_seed_gives_same_network():
    first = NeuralNetwork([4, 16, 7], seed=3)
    second = NeuralNetwork([4, 16, 7], seed=3)

    np.testing.assert_array_equal(
        first.predict([1, 0, 0, 1]), second.predict([1, 0, 0, 1])
    )


@pytest.mark.parametrize("layer_sizes", [[], [2], [2, 0, 1]])
def test_invalid_layer_sizes_are_rejected(layer_sizes):
    with pytest.raises(ValueError):
        NeuralNetwork(layer_sizes)


def test_wrong_input_or_target_size_is_rejected():
    network = NeuralNetwork([2, 2, 1], seed=0)
    with pytest.raises(ValueError):
        network.predict([1, 0, 1])
    with pytest.raises(ValueError):
        network.train([1, 0], [1, 0])


def test_xor_converges():
    network = NeuralNetwork([2, 2, 1], learning_rate=0.1, seed=0)

    losses = network.fit(XOR_INPUTS, XOR_TARGETS, epochs=20000)

    assert losses[-1] < losses[0]
    for inputs, targets in zip(XOR_INPUTS, XOR_TARGETS, strict=True):
        assert round(network.predict(inputs)[0]) == targets[0]


def test_seven_segment_decodes_all_digits():
    network = NeuralNetwork([4, 16, 7], learning_rate=0.1, seed=42)

    # Fewer epochs than the assignment's 20000 keep the test fast; already enough
    network.fit(SEVEN_SEGMENT_INPUTS, SEVEN_SEGMENT_TARGETS, epochs=5000)

    for inputs, targets in zip(
        SEVEN_SEGMENT_INPUTS, SEVEN_SEGMENT_TARGETS, strict=True
    ):
        np.testing.assert_array_equal(np.round(network.predict(inputs)), targets)
