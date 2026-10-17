"""Compare backpropagation against numerically estimated gradients."""

import copy

import numpy as np
import pytest

from neural_network import NeuralNetwork

EPSILON = 1e-6


def cost(network: NeuralNetwork, inputs, targets) -> float:
    outputs = network.predict(inputs)
    return 0.5 * float(np.sum((outputs - np.asarray(targets)) ** 2))


def analytic_gradients(network: NeuralNetwork, inputs, targets):
    # With alpha = 1 an update is w_new = w_old - dJ/dw, so the difference is
    # exactly the gradient backpropagation computed
    trained = copy.deepcopy(network)
    trained.learning_rate = 1.0
    trained.train(inputs, targets)

    gradients = []
    for old_layer, new_layer in zip(network.layers, trained.layers, strict=True):
        for old, new in zip(old_layer.neurons, new_layer.neurons, strict=True):
            gradients.append((old.weights - new.weights, old.bias - new.bias))
    return gradients


def numeric_gradients(network: NeuralNetwork, inputs, targets):
    gradients = []
    for layer in network.layers:
        for neuron in layer.neurons:
            weight_gradient = np.zeros_like(neuron.weights)
            for i in range(len(neuron.weights)):
                original = neuron.weights[i]
                neuron.weights[i] = original + EPSILON
                cost_plus = cost(network, inputs, targets)
                neuron.weights[i] = original - EPSILON
                cost_minus = cost(network, inputs, targets)
                neuron.weights[i] = original
                weight_gradient[i] = (cost_plus - cost_minus) / (2 * EPSILON)

            original = neuron.bias
            neuron.bias = original + EPSILON
            cost_plus = cost(network, inputs, targets)
            neuron.bias = original - EPSILON
            cost_minus = cost(network, inputs, targets)
            neuron.bias = original
            bias_gradient = (cost_plus - cost_minus) / (2 * EPSILON)

            gradients.append((weight_gradient, bias_gradient))
    return gradients


@pytest.mark.parametrize(
    ("layer_sizes", "inputs", "targets"),
    [
        ([2, 2, 1], [1, 0], [1]),
        ([4, 16, 7], [0, 1, 1, 0], [1, 0, 1, 1, 1, 1, 1]),
        ([3, 5, 4, 2], [0.2, -0.7, 1.5], [0.1, 0.9]),
    ],
)
def test_backpropagation_matches_numeric_gradient(layer_sizes, inputs, targets):
    network = NeuralNetwork(layer_sizes, seed=7)

    analytic = analytic_gradients(network, inputs, targets)
    numeric = numeric_gradients(network, inputs, targets)

    for (a_weights, a_bias), (n_weights, n_bias) in zip(analytic, numeric, strict=True):
        np.testing.assert_allclose(a_weights, n_weights, atol=1e-8)
        assert a_bias == pytest.approx(n_bias, abs=1e-8)
