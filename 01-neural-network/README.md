# Neural Network from Scratch

Feedforward neural network with backpropagation, written from scratch for INSY following the assignment "Neural Network from Scratch". Documentation: [website, assignment 01](https://jonas-wolfbauer.github.io/5AHITS_2026_2027_INSY/01-neural-network/). Two implementations of the same design:

- `python/`: functional implementation with numpy
- `csharp/`: the same structure in plain, single-threaded C# for readability

Both write byte-identical model files, so a network trained in one language can be loaded in the other.

## Structure

| Path | Content |
|---|---|
| `python/neural_network/` | `Neuron`, `Layer`, `NeuralNetwork`, activation functions, datasets, file formats, CLI |
| `python/neural_network/visualization/` | records training runs and renders the HTML page |
| `python/tests/` | gradient check, forward pass, convergence, persistence, cross-language files, page builder |
| `csharp/` | `Neuron`, `Layer`, `NeuralNetwork`, `NetworkSerializer`, datasets, CLI and built-in self-test |
| `models/python/`, `models/csharp/` | exported networks (`.json`, `.txt`, `.h`) |
| `visualization/` | generated page (not versioned) |

## Python

Requires Python 3.14. All commands run from `python/`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --no-cache-dir -r requirements.txt
```

Train the assignment networks (exit code 0 if every sample is predicted correctly):

```powershell
.venv\Scripts\python -m neural_network xor
.venv\Scripts\python -m neural_network xor --seed 8
.venv\Scripts\python -m neural_network seven-segment --export ..\models\python
.venv\Scripts\python -m neural_network --help
```

`--export DIR` writes `.json`, `.txt` and `.h`, reloads the JSON and TXT files and prints the largest prediction difference.

Build the visualization (about 20 s), then open `visualization/index.html` in a browser:

```powershell
.venv\Scripts\python -m neural_network visualize
```

Tests and style:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
.venv\Scripts\ruff format --check .
```

## C#

Requires the .NET 10 SDK; no NuGet packages. All commands run from `csharp/`.

```powershell
dotnet run -- xor
dotnet run -- xor --seed 1
dotnet run -- seven-segment --export ..\models\csharp
dotnet run -- load ..\models\python\xor.json
dotnet run -- --help
```

The options match the Python CLI (`--epochs`, `--lr`, `--seed`, `--hidden`, `--export`). `load` runs a model from either language on its dataset. Seeds differ from Python because `System.Random` is not numpy's generator.

Self-test (the checks of the Python tests, plus loading the Python models and comparing the files they produce byte by byte) and style:

```powershell
dotnet run -- selftest
dotnet format --verify-no-changes
```

The build enforces the `.editorconfig` rules (braces, naming) and treats warnings as errors.

## File formats

| Format | Use |
|---|---|
| `.json` | full network, `{"format": "insy-nn", "version": 1, "activation", "input_size", "layers": [{"neurons": [{"weights", "bias"}]}]}` |
| `.txt` | human-readable listing, every value preceded by a keyword; parseable with `fscanf` |
| `.h` | C constants: `<PREFIX>_INPUT_SIZE`, `<PREFIX>_LAYER_COUNT`, `<PREFIX>_LAYER_SIZES`, `<PREFIX>_L<n>_WEIGHTS`, `<PREFIX>_L<n>_BIASES` |

Floats are written with full round-trip precision in Python's `repr` notation (C# reproduces it in `NetworkSerializer.FormatFloat`), so a reload predicts bit-identically and both languages produce the same bytes.

## Status

- [x] Part 1 in Python and C#: network, training, JSON/TXT export and import
- [x] Part 1, step 6: C header (constants only; the init function follows with the C structs of part 4)
- [x] Part 2 in Python and C#: XOR and 7-segment trained, exported, reloaded; models load across languages
- [x] Visualization page (Python)
