# Neural Network from Scratch

Feedforward neural network with backpropagation, written from scratch for INSY following the assignment "Neural Network from Scratch". Documentation: [website, assignment 01](https://jonas-wolfbauer.github.io/5AHITS_2026_2027_INSY/01-neural-network/). Two implementations of the same design:

- `python/`: functional implementation with numpy
- `csharp/`: the same structure in plain, single-threaded C# for readability

Both write byte-identical model files, so a network trained in one language can be loaded in the other. On top of that:

- `tensorflow/`: the same networks trained with TensorFlow, compared with the own implementation (part 3)
- `c/`: forward-only inference in C, weights from the `.txt` file or the generated `.h` header (parts 4 and 5)

## Structure

| Path | Content |
|---|---|
| `python/neural_network/` | `Neuron`, `Layer`, `NeuralNetwork`, activation functions, datasets, file formats, CLI |
| `python/neural_network/visualization/` | records training runs and renders the HTML page |
| `python/tests/` | gradient check, forward pass, convergence, persistence, cross-language files, page builder |
| `csharp/` | `Neuron`, `Layer`, `NeuralNetwork`, `NetworkSerializer`, datasets, CLI and built-in self-test |
| `tensorflow/tf_comparison/` | Keras models, SGD training loop, comparison CLI; tests in `tensorflow/tests/` |
| `c/` | `neural_network.h/.c` (structs, `.txt` loader, forward pass), `predict_txt.c`, `predict_header.c`, `selftest.c`, `Makefile` |
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

## TensorFlow comparison (part 3)

TensorFlow 2.21 has no packages for Python 3.14, so `tensorflow/` gets its own environment with Python 3.12. With [uv](https://docs.astral.sh/uv/), from the repo root (the interpreter lands in `.tools/`, not versioned):

```powershell
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.tools\python"
uv python install 3.12 --no-bin --no-registry
uv venv --python 3.12 01-neural-network\tensorflow\.venv
uv pip install --python 01-neural-network\tensorflow\.venv\Scripts\python.exe -r 01-neural-network\tensorflow\requirements.txt
```

Then from `tensorflow/` (imports the own implementation from `../python`; Keras keeps its settings in `tensorflow/.keras/`):

```powershell
.venv\Scripts\python -m tf_comparison
.venv\Scripts\python -m tf_comparison xor --epochs 5000
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
```

Every problem is trained three times: own implementation, TensorFlow from the same initial weights (must follow the same path; checks the own backpropagation) and TensorFlow with Keras' default initialization. Exit code 0 if the same-start runs agree to 1e-6.

## C inference (parts 4 and 5)

Needs `gcc` and `make`. A portable toolchain without installation is [w64devkit](https://github.com/skeeto/w64devkit) (v2.10.0, SHA-256 `18d0a4c7…3c6325e`), extracted to `.tools/w64devkit/`. From `c/`:

```powershell
$env:PATH = "$PWD\..\..\.tools\w64devkit\bin;$env:PATH"
make test
make MODELS=../models/csharp test
```

`make test` runs the self-test (forward pass, rejected files, `.txt` and `.h` give bit-identical outputs), `predict_txt` on both `.txt` models (part 4) and `predict_header` (part 5, no file and no `malloc`). The generated headers include `neural_network.h`, so `c/` must be on the include path.

## File formats

| Format | Use |
|---|---|
| `.json` | full network, `{"format": "insy-nn", "version": 1, "activation", "input_size", "layers": [{"neurons": [{"weights", "bias"}]}]}` |
| `.txt` | human-readable listing, every value preceded by a keyword; parseable with `fscanf` |
| `.h` | C constants `<PREFIX>_INPUT_SIZE`, `<PREFIX>_LAYER_COUNT`, `<PREFIX>_LAYER_SIZES`, `<PREFIX>_L<n>_WEIGHTS`, `<PREFIX>_L<n>_BIASES` and `<PREFIX>_Init(Network *)`, which builds the structs of `c/neural_network.h` around them |

Floats are written with full round-trip precision in Python's `repr` notation (C# reproduces it in `NetworkSerializer.FormatFloat`), so a reload predicts bit-identically and both languages produce the same bytes.

## Status

- [x] Part 1 in Python and C#: network, training, JSON/TXT export and import, C header with init function
- [x] Part 2 in Python and C#: XOR and 7-segment trained, exported, reloaded; models load across languages
- [x] Part 3: comparison with TensorFlow
- [x] Parts 4 and 5: forward-only inference in C from `.txt` and from the header
- [ ] Part 6 (ESP32): not in scope
- [x] Visualization page (Python)
