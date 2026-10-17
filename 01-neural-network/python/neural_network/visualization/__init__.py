"""Record training runs and render them into a self-contained HTML page.

The page does not compute anything itself: every weight, activation and cost
it shows comes from this package's ``NeuralNetwork`` during training.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neural_network.datasets import (
    SEVEN_SEGMENT_INPUTS,
    SEVEN_SEGMENT_TARGETS,
    XOR_INPUTS,
    XOR_TARGETS,
)
from neural_network.network import NeuralNetwork

LEARNING_RATE = 0.1
EPOCHS = 20000
SNAPSHOT_COUNT = 90
COST_CURVE_POINTS = 300
GRID_SIZE = 25

_TEMPLATE = Path(__file__).with_name("template.html")
_DATA_PLACEHOLDER = "__NN_DATA__"


@dataclass(frozen=True)
class Scenario:
    """One training run shown on the page."""

    id: str
    kind: str
    title: str
    description: str
    layer_sizes: list[int]
    seed: int
    inputs: list[list[int]]
    targets: list[list[int]]


SCENARIOS = (
    Scenario(
        id="xor",
        kind="xor",
        title="XOR · Seed 0",
        description=(
            "2 Inputs, 2 Hidden-Neuronen, 1 Output. Das Netz lernt XOR: Ein "
            "Hidden-Neuron erkennt »x₁ und nicht x₂«, das andere »x₂ und nicht "
            "x₁«, der Output verknüpft beide zu einem OR."
        ),
        layer_sizes=[2, 2, 1],
        seed=0,
        inputs=XOR_INPUTS,
        targets=XOR_TARGETS,
    ),
    Scenario(
        id="xor-stuck",
        kind="xor",
        title="XOR · Seed 8 (lokales Minimum)",
        description=(
            "Gleiches Netz, andere Startgewichte. Das Training bleibt in einem "
            "lokalen Minimum hängen: [0, 1] wird gelernt, für [1, 0] und [1, 1] "
            "landet der Output bei etwa 0,5. Bei Seeds 0–19 passiert das in "
            "4 von 20 Läufen."
        ),
        layer_sizes=[2, 2, 1],
        seed=8,
        inputs=XOR_INPUTS,
        targets=XOR_TARGETS,
    ),
    Scenario(
        id="seven-segment",
        kind="seven-segment",
        title="7-Segment-Decoder · Seed 42",
        description=(
            "4 Inputs (BCD-Ziffer 0–9), 16 Hidden-Neuronen, 7 Outputs "
            "(Segmente a–g). Die Helligkeit eines Segments ist der Output des "
            "zugehörigen Neurons."
        ),
        layer_sizes=[4, 16, 7],
        seed=42,
        inputs=SEVEN_SEGMENT_INPUTS,
        targets=SEVEN_SEGMENT_TARGETS,
    ),
)


def build_page(
    output: str | Path, progress: Callable[[str], None] | None = None
) -> None:
    """Train every scenario and write the HTML page.

    Parameters
    ----------
    output : str | Path
        Target ``.html`` file; parent folders are created.
    progress : Callable[[str], None] | None
        Receives one status line per scenario, e.g. ``print``.
    """
    scenarios = []
    for scenario in SCENARIOS:
        if progress is not None:
            progress(f"training {scenario.title} ...")
        scenarios.append(record(scenario))

    data = {
        "learningRate": LEARNING_RATE,
        "epochs": EPOCHS,
        "gridSize": GRID_SIZE,
        "scenarios": scenarios,
    }
    # "</" inside the embedded JSON would end the <script> element early
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    payload = payload.replace("</", "<\\/")

    html = _TEMPLATE.read_text(encoding="utf-8").replace(_DATA_PLACEHOLDER, payload)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8", newline="\n")


def record(scenario: Scenario) -> dict[str, Any]:
    """Train one scenario and capture snapshots along the way.

    Parameters
    ----------
    scenario : Scenario
        Network layout, seed and dataset.

    Returns
    -------
    dict[str, Any]
        Scenario metadata, cost curve and snapshots for the page.
    """
    network = NeuralNetwork(scenario.layer_sizes, LEARNING_RATE, scenario.seed)
    snapshot_epochs = set(_log_spaced_epochs(EPOCHS, SNAPSHOT_COUNT))

    initial_cost = network.mean_cost(scenario.inputs, scenario.targets)
    snapshots = [_snapshot(network, scenario, 0)]

    def on_epoch(epoch: int, _cost: float) -> None:
        if epoch in snapshot_epochs:
            snapshots.append(_snapshot(network, scenario, epoch))

    costs = network.fit(scenario.inputs, scenario.targets, EPOCHS, on_epoch=on_epoch)

    curve_epochs = _log_spaced_epochs(EPOCHS, COST_CURVE_POINTS)
    cost_curve = [[0, _significant(initial_cost)]] + [
        [epoch, _significant(costs[epoch - 1])] for epoch in curve_epochs
    ]

    return {
        "id": scenario.id,
        "kind": scenario.kind,
        "title": scenario.title,
        "description": scenario.description,
        "layerSizes": scenario.layer_sizes,
        "seed": scenario.seed,
        "inputs": scenario.inputs,
        "targets": scenario.targets,
        "costCurve": cost_curve,
        "snapshots": snapshots,
    }


def _snapshot(network: NeuralNetwork, scenario: Scenario, epoch: int) -> dict[str, Any]:
    activations = []
    for sample in scenario.inputs:
        network.predict(sample)
        # Every neuron keeps its last output, which is exactly what is drawn
        activations.append(
            [[round(n.output, 3) for n in layer.neurons] for layer in network.layers]
        )

    snapshot: dict[str, Any] = {
        "epoch": epoch,
        "cost": _significant(network.mean_cost(scenario.inputs, scenario.targets)),
        "layers": [
            {
                "weights": [
                    [round(float(w), 4) for w in neuron.weights]
                    for neuron in layer.neurons
                ],
                "biases": [round(neuron.bias, 4) for neuron in layer.neurons],
            }
            for layer in network.layers
        ],
        "activations": activations,
    }

    if scenario.kind == "xor":
        axis = np.linspace(0.0, 1.0, GRID_SIZE)
        # Row-major, first row is x2 = 0
        snapshot["grid"] = [
            round(float(network.predict([x1, x2])[0]), 3) for x2 in axis for x1 in axis
        ]
    return snapshot


def _log_spaced_epochs(epochs: int, count: int) -> list[int]:
    # Dense at the start, where the network changes fastest
    points = np.unique(np.round(np.geomspace(1, epochs, count)).astype(int))
    return [int(p) for p in points]


def _significant(value: float) -> float:
    return float(f"{value:.4g}")
