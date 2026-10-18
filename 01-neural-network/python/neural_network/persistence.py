"""Text formats for networks: JSON, human-readable TXT and C header.

All functions work on the plain dict produced by ``NeuralNetwork.to_dict``::

    {
        "format": "insy-nn",
        "version": 1,
        "activation": "sigmoid",
        "input_size": 2,
        "layers": [{"neurons": [{"weights": [...], "bias": 0.1}, ...]}, ...],
    }

The C# implementation reads and writes exactly the same formats.
"""

import json
import re
from collections.abc import Iterator
from typing import Any

FORMAT_NAME = "insy-nn"
FORMAT_VERSION = 1

type NetworkDict = dict[str, Any]

_C_IDENTIFIER = re.compile(r"^[A-Z_][A-Z0-9_]*$")


def validate(data: NetworkDict) -> None:
    """Check that a network dict is complete and its shapes fit together.

    Parameters
    ----------
    data : NetworkDict
        Network in the shared dict layout.

    Raises
    ------
    ValueError
        If format, version or any shape is wrong.
    """
    if data.get("format") != FORMAT_NAME or data.get("version") != FORMAT_VERSION:
        raise ValueError(f"not an {FORMAT_NAME} v{FORMAT_VERSION} network")
    if not data.get("layers"):
        raise ValueError("network has no layers")

    expected_inputs = data["input_size"]
    for layer_number, layer in enumerate(data["layers"], start=1):
        if not layer["neurons"]:
            raise ValueError(f"layer {layer_number} has no neurons")
        for neuron in layer["neurons"]:
            if len(neuron["weights"]) != expected_inputs:
                raise ValueError(
                    f"layer {layer_number}: expected {expected_inputs} weights "
                    f"per neuron, got {len(neuron['weights'])}"
                )
        expected_inputs = len(layer["neurons"])


def to_json(data: NetworkDict) -> str:
    """Serialize a network dict to JSON.

    Parameters
    ----------
    data : NetworkDict
        Network in the shared dict layout.

    Returns
    -------
    str
        Indented JSON; floats keep full precision.
    """
    return json.dumps(data, indent=2) + "\n"


def from_json(text: str) -> NetworkDict:
    """Parse and validate JSON produced by ``to_json``.

    Parameters
    ----------
    text : str
        JSON document.

    Returns
    -------
    NetworkDict
        Validated network dict.
    """
    data = json.loads(text)
    validate(data)
    return data


def to_txt(data: NetworkDict) -> str:
    """Format a network as a human-readable, whitespace-separated text file.

    Every value is preceded by a keyword, so the file reads like a listing
    and a C program can parse it with ``fscanf`` token by token.

    Parameters
    ----------
    data : NetworkDict
        Network in the shared dict layout.

    Returns
    -------
    str
        Text representation.
    """
    lines = [
        f"{FORMAT_NAME} {FORMAT_VERSION}",
        f"activation {data['activation']}",
        f"input_size {data['input_size']}",
        f"layers {len(data['layers'])}",
    ]

    input_count = data["input_size"]
    for layer_number, layer in enumerate(data["layers"], start=1):
        neurons = layer["neurons"]
        lines.append("")
        lines.append(
            f"layer {layer_number} neurons {len(neurons)} inputs {input_count}"
        )
        for neuron_number, neuron in enumerate(neurons, start=1):
            weights = " ".join(_format_float(w) for w in neuron["weights"])
            lines.append(
                f"neuron {neuron_number} bias {_format_float(neuron['bias'])} "
                f"weights {weights}"
            )
        input_count = len(neurons)

    return "\n".join(lines) + "\n"


def from_txt(text: str) -> NetworkDict:
    """Parse a file produced by ``to_txt``.

    Parameters
    ----------
    text : str
        Content of the text file.

    Returns
    -------
    NetworkDict
        Validated network dict.
    """
    tokens = iter(text.split())

    _expect(tokens, FORMAT_NAME)
    version = int(next(tokens))
    _expect(tokens, "activation")
    activation = next(tokens)
    _expect(tokens, "input_size")
    input_size = int(next(tokens))
    _expect(tokens, "layers")
    layer_count = int(next(tokens))

    layers = []
    for layer_number in range(1, layer_count + 1):
        _expect(tokens, "layer", str(layer_number), "neurons")
        neuron_count = int(next(tokens))
        _expect(tokens, "inputs")
        input_count = int(next(tokens))

        neurons = []
        for neuron_number in range(1, neuron_count + 1):
            _expect(tokens, "neuron", str(neuron_number), "bias")
            bias = float(next(tokens))
            _expect(tokens, "weights")
            weights = [float(next(tokens)) for _ in range(input_count)]
            neurons.append({"weights": weights, "bias": bias})
        layers.append({"neurons": neurons})

    data = {
        "format": FORMAT_NAME,
        "version": version,
        "activation": activation,
        "input_size": input_size,
        "layers": layers,
    }
    validate(data)
    return data


def to_c_header(data: NetworkDict, prefix: str) -> str:
    """Generate a C header with all constants needed to rebuild the network.

    Parameters
    ----------
    data : NetworkDict
        Network in the shared dict layout.
    prefix : str
        Name prefix for every symbol, e.g. ``"XOR"``; lets several networks
        live in one C program.

    Returns
    -------
    str
        Content of the ``.h`` file.
    """
    prefix = prefix.upper()
    if not _C_IDENTIFIER.match(prefix):
        raise ValueError(f"prefix {prefix!r} is not a valid C identifier")

    layers = data["layers"]
    sizes = ", ".join(str(len(layer["neurons"])) for layer in layers)
    guard = f"{prefix}_NET_H"

    lines = [
        f"// Generated by neural_network ({FORMAT_NAME} v{FORMAT_VERSION}). "
        "Do not edit.",
        f"// Activation: {data['activation']}",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "neural_network.h"',
        "",
        f"#define {prefix}_INPUT_SIZE {data['input_size']}",
        f"#define {prefix}_LAYER_COUNT {len(layers)}",
        "",
        f"static const int {prefix}_LAYER_SIZES[{prefix}_LAYER_COUNT] = {{{sizes}}};",
    ]

    input_count = data["input_size"]
    for layer_number, layer in enumerate(layers, start=1):
        neurons = layer["neurons"]
        symbol = f"{prefix}_L{layer_number}"
        lines.append("")
        lines.append(
            f"// Layer {layer_number}: {len(neurons)} neurons x {input_count} inputs"
        )
        lines.append(
            f"static const double {symbol}_WEIGHTS[{len(neurons)}][{input_count}] = {{"
        )
        for neuron in neurons:
            weights = ", ".join(_format_float(w) for w in neuron["weights"])
            lines.append(f"    {{{weights}}},")
        lines.append("};")
        biases = ", ".join(_format_float(n["bias"]) for n in neurons)
        lines.append(
            f"static const double {symbol}_BIASES[{len(neurons)}] = {{{biases}}};"
        )
        input_count = len(neurons)

    lines.append("")
    lines.extend(_c_init_function(data, prefix))
    lines.append("")
    lines.append(f"#endif // {guard}")
    return "\n".join(lines) + "\n"


def _c_init_function(data: NetworkDict, prefix: str) -> list[str]:
    # Builds the structs of c/neural_network.h around the constant arrays: no heap
    # memory and no file, as needed on a microcontroller
    layers = data["layers"]
    lines = [
        f"// Storage for the network that {prefix}_Init builds; no heap memory needed"
    ]
    for layer_number, layer in enumerate(layers, start=1):
        count = len(layer["neurons"])
        lines.append(f"static Neuron {prefix}_L{layer_number}_NEURONS[{count}];")
    lines.append(f"static Layer {prefix}_LAYERS[{prefix}_LAYER_COUNT];")
    lines.append("")
    lines.append("/// Builds the network from the constants above.")
    lines.append(
        "/// The weights stay in the constant arrays: never pass it to NetworkFree."
    )
    lines.append(f"static inline void {prefix}_Init (Network *network)")
    lines.append("{")

    input_count = data["input_size"]
    for index, layer in enumerate(layers):
        count = len(layer["neurons"])
        symbol = f"{prefix}_L{index + 1}"
        lines.append(f"    for (int n = 0; n < {count}; n++)")
        lines.append("    {")
        lines.append(
            f"        {symbol}_NEURONS[n] = (Neuron){{.weights = {symbol}_WEIGHTS[n], "
            f".bias = {symbol}_BIASES[n], .inputCount = {input_count}}};"
        )
        lines.append("    }")
        lines.append(
            f"    {prefix}_LAYERS[{index}] = (Layer){{.neurons = {symbol}_NEURONS, "
            f".neuronCount = {count}, .inputCount = {input_count}}};"
        )
        input_count = count

    lines.append(
        f"    *network = (Network){{.layers = {prefix}_LAYERS, "
        f".layerCount = {prefix}_LAYER_COUNT, .inputSize = {prefix}_INPUT_SIZE}};"
    )
    lines.append("}")
    return lines


def _format_float(value: float) -> str:
    # repr is the shortest string that parses back to the identical double
    return repr(float(value))


def _expect(tokens: Iterator[str], *expected: str) -> None:
    for keyword in expected:
        token = next(tokens, None)
        if token != keyword:
            raise ValueError(f"expected {keyword!r}, got {token!r}")
