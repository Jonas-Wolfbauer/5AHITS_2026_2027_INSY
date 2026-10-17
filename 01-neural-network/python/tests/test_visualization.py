"""The page builder embeds valid data for every scenario."""

import json
import re

from neural_network import visualization


def test_page_embeds_all_scenarios(tmp_path, monkeypatch):
    # A short run is enough to check structure; the real page uses 20000 epochs
    monkeypatch.setattr(visualization, "EPOCHS", 30)
    monkeypatch.setattr(visualization, "SNAPSHOT_COUNT", 10)
    monkeypatch.setattr(visualization, "COST_CURVE_POINTS", 10)
    output = tmp_path / "page" / "index.html"

    visualization.build_page(output)

    html = output.read_text(encoding="utf-8")
    assert "__NN_DATA__" not in html
    payload = re.search(
        r'<script id="nn-data" type="application/json">(.*?)</script>', html, re.S
    )
    data = json.loads(payload.group(1))

    assert [s["id"] for s in data["scenarios"]] == ["xor", "xor-stuck", "seven-segment"]
    for scenario in data["scenarios"]:
        snapshots = scenario["snapshots"]
        assert snapshots[0]["epoch"] == 0
        assert snapshots[-1]["epoch"] == 30
        assert scenario["costCurve"][-1][0] == 30
        layer_sizes = scenario["layerSizes"]
        for snapshot in snapshots:
            assert len(snapshot["activations"]) == len(scenario["inputs"])
            assert [len(layer["biases"]) for layer in snapshot["layers"]] == (
                layer_sizes[1:]
            )
            if scenario["kind"] == "xor":
                assert len(snapshot["grid"]) == visualization.GRID_SIZE**2


def test_snapshot_epochs_are_dense_at_the_start():
    epochs = visualization._log_spaced_epochs(20000, 90)

    assert epochs[0] == 1
    assert epochs[-1] == 20000
    assert epochs == sorted(set(epochs))
    assert sum(epoch <= 100 for epoch in epochs) > sum(
        epoch > 10000 for epoch in epochs
    )
