"""TensorFlow from the same start must compute what the own implementation does."""

import numpy as np

from neural_network import NeuralNetwork
from tf_comparison.comparison import (
    PROBLEMS,
    build_model,
    compare,
    copy_weights,
    train_tensorflow,
)


def keras_parameters(model):
    # Back to the own layout: one row of weights per neuron
    return [
        (kernel.T, biases)
        for kernel, biases in (dense.get_weights() for dense in model.layers)
    ]


def own_parameters(network):
    return [
        (
            np.array([neuron.weights for neuron in layer.neurons]),
            np.array([neuron.bias for neuron in layer.neurons]),
        )
        for layer in network.layers
    ]


def test_copied_weights_predict_the_same():
    network = NeuralNetwork([3, 5, 4, 2], seed=7)
    model = build_model([3, 5, 4, 2])
    copy_weights(network, model)

    sample = [0.2, -0.7, 1.5]
    keras_output = model(np.array([sample]), training=False).numpy()[0]

    np.testing.assert_allclose(keras_output, network.predict(sample), atol=1e-15)


def test_one_sgd_step_matches_the_own_update():
    network = NeuralNetwork([3, 5, 4, 2], seed=7)
    model = build_model([3, 5, 4, 2])
    copy_weights(network, model)

    network.train([0.2, -0.7, 1.5], [0.1, 0.9])
    train_tensorflow(model, [[0.2, -0.7, 1.5]], [[0.1, 0.9]], epochs=1)

    for (own_w, own_b), (tf_w, tf_b) in zip(
        own_parameters(network), keras_parameters(model), strict=True
    ):
        np.testing.assert_allclose(tf_w, own_w, atol=1e-14)
        np.testing.assert_allclose(tf_b, own_b, atol=1e-14)


def test_same_start_follows_the_own_training():
    comparison = compare(PROBLEMS["xor"], epochs=500)

    np.testing.assert_allclose(
        comparison.same_start.costs, comparison.own.costs, atol=1e-12
    )
    # Rounding in a different order adds up to about 1e-10; a wrong gradient
    # would differ by orders of magnitude more
    assert comparison.max_difference < 1e-8
