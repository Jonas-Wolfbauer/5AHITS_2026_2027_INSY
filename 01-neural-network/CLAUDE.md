# CLAUDE.md

INSY assignment 01: neural network from scratch in Python and C#, following the assignment *4AHITS_NN_from_scratch* (parts 1 and 2 in scope). Repo-wide context and constraints: `../CLAUDE.md`.

## Context

- Project note, decisions and progress: school vault `INSY/01-neural-network/_overview.md`.
- Documentation pages: school vault `INSY/01-neural-network/`.

## Design

- `NeuralNetwork(layer_sizes, learning_rate, seed)`: the first entry of `layer_sizes` is the input size; the input is not a layer with weights.
- Per-neuron structure as required by the assignment (`Neuron` → `Layer` → `NeuralNetwork`), sigmoid activation, cost J = 1/2 · Σ(ŷ − y)², SGD with one update per sample.
- `Layer.backward` must compute the error for the previous layer with the weights from the forward pass, before updating them.
- Python and C# share the same API and file formats.
- Text formats (`persistence.py`) only convert the dict from `NeuralNetwork.to_dict`; the C# side must produce byte-compatible files.
- The visualization page never computes the network itself; all values are recorded from Python during training. `insy-site.py` publishes `visualization/` as `visualisierung/` on the website, so regenerate it before a publish.

## Run and test

See `README.md` for all commands. Quick check from `python/`:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
```
