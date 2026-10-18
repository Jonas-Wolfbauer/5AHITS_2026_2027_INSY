"""Export and import: JSON, TXT and C header."""

import json
import re

import numpy as np
import pytest

from neural_network import NeuralNetwork, persistence

SAMPLES = [[0, 0, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0.3, -2.0, 7.5, 1.0]]


@pytest.fixture
def network() -> NeuralNetwork:
    return NeuralNetwork([4, 5, 3], seed=11)


def assert_same_predictions(first: NeuralNetwork, second: NeuralNetwork):
    for sample in SAMPLES:
        np.testing.assert_array_equal(first.predict(sample), second.predict(sample))


def test_json_round_trip_is_exact(network, tmp_path):
    path = tmp_path / "net.json"

    network.dump_json(path)
    loaded = NeuralNetwork.load(path)

    assert loaded.layer_sizes == network.layer_sizes
    assert_same_predictions(network, loaded)


def test_json_has_shared_layout(network, tmp_path):
    path = tmp_path / "net.json"
    network.dump_json(path)

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["format"] == "insy-nn"
    assert data["version"] == 1
    assert data["activation"] == "sigmoid"
    assert data["input_size"] == 4
    assert [len(layer["neurons"]) for layer in data["layers"]] == [5, 3]
    assert len(data["layers"][1]["neurons"][0]["weights"]) == 5


def test_txt_round_trip_is_exact(network, tmp_path):
    path = tmp_path / "net.txt"

    network.dump_txt(path)
    loaded = NeuralNetwork.load_txt(path)

    assert_same_predictions(network, loaded)


def test_txt_is_readable(network):
    lines = persistence.to_txt(network.to_dict()).splitlines()

    assert lines[:4] == ["insy-nn 1", "activation sigmoid", "input_size 4", "layers 2"]
    assert "layer 1 neurons 5 inputs 4" in lines
    assert "layer 2 neurons 3 inputs 5" in lines
    assert lines[6].startswith("neuron 1 bias ")


def test_c_header_contains_every_parameter(network):
    data = network.to_dict()

    header = persistence.to_c_header(data, "demo")

    assert "#ifndef DEMO_NET_H" in header
    assert "#define DEMO_INPUT_SIZE 4" in header
    assert "#define DEMO_LAYER_COUNT 2" in header
    assert "DEMO_LAYER_SIZES[DEMO_LAYER_COUNT] = {5, 3};" in header
    assert "DEMO_L1_WEIGHTS[5][4]" in header
    assert "DEMO_L2_BIASES[3]" in header
    assert '#include "neural_network.h"' in header
    assert "static inline void DEMO_Init (Network *network)" in header

    # Every number in the arrays must parse back to the exact stored double
    body = header.split("DEMO_LAYER_SIZES", 1)[1].split("};", 1)[1]
    numbers = [
        float(token) for token in re.findall(r"-?\d+\.\d+(?:e-?\d+)?|-?\d+e-?\d+", body)
    ]
    expected = []
    for layer in data["layers"]:
        for neuron in layer["neurons"]:
            expected.extend(neuron["weights"])
        expected.extend(neuron["bias"] for neuron in layer["neurons"])
    assert numbers == expected


def test_invalid_c_prefix_is_rejected(network):
    with pytest.raises(ValueError):
        persistence.to_c_header(network.to_dict(), "7seg")


def test_broken_files_are_rejected(network):
    data = network.to_dict()
    data["layers"][1]["neurons"][0]["weights"].pop()
    with pytest.raises(ValueError):
        NeuralNetwork.from_dict(data)

    with pytest.raises(ValueError):
        persistence.from_json('{"format": "other", "version": 1}')

    with pytest.raises(ValueError):
        persistence.from_txt("insy-nn 1\nactivation sigmoid\ninput_size 4\nlayer 1")
