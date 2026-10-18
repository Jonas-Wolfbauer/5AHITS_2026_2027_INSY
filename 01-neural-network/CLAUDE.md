# CLAUDE.md

INSY assignment 01: neural network from scratch in Python and C#, following the assignment "Neural Network from Scratch" (parts 1 and 2 in scope). Repo-wide context and constraints: `../CLAUDE.md`.

## Context

- Project note, decisions and progress: school vault `INSY/01-neural-network/_overview.md`.
- Documentation pages: school vault `INSY/01-neural-network/`.

## Design

- `NeuralNetwork(layer_sizes, learning_rate, seed)`: the first entry of `layer_sizes` is the input size; the input is not a layer with weights.
- Per-neuron structure as required by the assignment (`Neuron` → `Layer` → `NeuralNetwork`), sigmoid activation, cost J = 1/2 · Σ(ŷ − y)², SGD with one update per sample.
- `Layer.backward` must compute the error for the previous layer with the weights from the forward pass, before updating them.
- Python and C# share the same API and file formats.
- Text formats (`persistence.py`) only convert the dict from `NeuralNetwork.to_dict`; the C# side (`NetworkSerializer`, `NetworkData`) produces byte-identical files. Floats follow Python's `repr` (`NetworkSerializer.FormatFloat`); both test suites compare the committed models of the other language byte by byte.
- C# initial weights come from `System.Random`, so seeds differ from Python (XOR: seeds 1 and 6 of 0–19 get stuck in C#, 8, 12, 13 and 15 in Python).
- The `.h` header contains `<PREFIX>_Init(Network *)` for the structs in `c/neural_network.h`; Python (`persistence._c_init_function`) and C# (`NetworkSerializer.CInitFunction`) must generate it identically. Changing the header format means regenerating all four `.h` files in `models/`.
- `tensorflow/` (part 3) runs on Python 3.12 (TensorFlow has no 3.14 packages) and imports `../python`; that is why `neuron.py` and `network.py` use `from __future__ import annotations`. Its cost is J = 1/2 · Σ(ŷ − y)², not Keras' MSE.
- `c/` (parts 4 and 5) is forward-only C17; C naming per the conventions (functions PascalCase, constants UPPER_SNAKE_CASE, files snake_case).
- The visualization page never computes the network itself; all values are recorded from Python during training. `insy-site.py` publishes `visualization/` as `visualisierung/` on the website, so regenerate it before a publish.

## Run and test

See `README.md` for all commands. Quick check from `python/`:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
```

From `csharp/`:

```powershell
dotnet build --disable-build-servers
dotnet run -- selftest
dotnet format --verify-no-changes
```

Before running dotnet, point everything it would write into the user profile to `.tmp/` in the repo root (and delete `.tmp/` afterwards): `TEMP`, `TMP`, `DOTNET_CLI_HOME`, `NUGET_PACKAGES`, `NUGET_HTTP_CACHE_PATH`, plus `DOTNET_NOLOGO=1` and `DOTNET_CLI_TELEMETRY_OPTOUT=1`. Without `DOTNET_CLI_HOME`, the first run of a new SDK version writes sentinel files to `%USERPROFILE%\.dotnet`.

From `tensorflow/` (Python 3.12 from `.tools/python`, installed with `UV_PYTHON_INSTALL_DIR` and `--no-bin --no-registry`; `tf_comparison` sets `KERAS_HOME` to `tensorflow/.keras`):

```powershell
.venv\Scripts\python -m pytest -p no:cacheprovider
.venv\Scripts\ruff check .
.venv\Scripts\python -m tf_comparison
```

From `c/`, with `.tools\w64devkit\bin` (GCC, make) first on `PATH`:

```powershell
make test
make MODELS=../models/csharp test
```
