"""Activation functions and their derivatives."""

import math
from collections.abc import Callable

type ActivationFn = Callable[[float], float]


def sigmoid(x: float) -> float:
    """Apply the logistic sigmoid.

    Parameters
    ----------
    x : float
        Pre-activation value z.

    Returns
    -------
    float
        Value in the open interval (0, 1).
    """
    # Two branches so exp never receives a large positive argument (overflow)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def sigmoid_derivative(output: float) -> float:
    """Compute the sigmoid derivative from the sigmoid's own output.

    Parameters
    ----------
    output : float
        Already activated value a = sigmoid(z), not z itself.

    Returns
    -------
    float
        da/dz = a * (1 - a).
    """
    return output * (1.0 - output)


# Name -> (function, derivative); the name is what gets stored in exported files
ACTIVATIONS: dict[str, tuple[ActivationFn, ActivationFn]] = {
    "sigmoid": (sigmoid, sigmoid_derivative),
}
