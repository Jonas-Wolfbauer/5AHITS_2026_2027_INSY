"""Networks exported by Python and C# load in Python, and both write identical files."""

from pathlib import Path

import numpy as np
import pytest

from neural_network import NeuralNetwork, persistence
from neural_network.datasets import (
    SEVEN_SEGMENT_INPUTS,
    SEVEN_SEGMENT_TARGETS,
    XOR_INPUTS,
    XOR_TARGETS,
)

MODELS = Path(__file__).resolve().parents[2] / "models"


@pytest.mark.parametrize("language", ["python", "csharp"])
@pytest.mark.parametrize("suffix", [".json", ".txt"])
def test_exported_networks_predict_correctly(language, suffix):
    load = NeuralNetwork.load if suffix == ".json" else NeuralNetwork.load_txt

    xor = load(MODELS / language / f"xor{suffix}")
    for inputs, targets in zip(XOR_INPUTS, XOR_TARGETS, strict=True):
        assert round(xor.predict(inputs)[0]) == targets[0]

    seven_segment = load(MODELS / language / f"seven_segment{suffix}")
    for inputs, targets in zip(
        SEVEN_SEGMENT_INPUTS, SEVEN_SEGMENT_TARGETS, strict=True
    ):
        np.testing.assert_array_equal(np.round(seven_segment.predict(inputs)), targets)


@pytest.mark.parametrize("name", ["xor", "seven_segment"])
def test_files_from_csharp_models_are_byte_identical(name):
    directory = MODELS / "csharp"
    data = persistence.from_json(read(directory / f"{name}.json"))

    assert persistence.to_json(data) == read(directory / f"{name}.json")
    assert persistence.to_txt(data) == read(directory / f"{name}.txt")
    assert persistence.to_c_header(data, name) == read(directory / f"{name}.h")


def read(path: Path) -> str:
    # read_text turns CRLF from a Windows checkout into LF, as used by the formats
    return path.read_text(encoding="utf-8")
