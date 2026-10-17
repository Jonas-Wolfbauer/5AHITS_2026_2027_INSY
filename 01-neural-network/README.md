# Neural Network from Scratch

Feedforward neural network with backpropagation, written from scratch for INSY following the assignment *4AHITS_NN_from_scratch*. Documentation: [website, assignment 01](https://jonas-wolfbauer.github.io/5AHITS_2026_2027_INSY/01-neural-network/). Two implementations of the same design:

- `python/`: functional implementation with numpy
- `csharp/`: the same structure in plain, single-threaded C# for readability

Both use the same model file formats, so a network trained in one language can be loaded in the other.

## Structure

| Path | Content |
|---|---|
| `python/neural_network/` | `Neuron`, `Layer`, `NeuralNetwork`, activation functions, datasets, file formats, CLI |
| `python/neural_network/visualization/` | records training runs and renders the HTML page |
| `python/tests/` | gradient check, forward pass, convergence, persistence, page builder |
| `csharp/` | C# implementation and self-test |
| `models/` | exported networks (`.json`, `.txt`, `.h`) |
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

## File formats

| Format | Use |
|---|---|
| `.json` | full network, `{"format": "insy-nn", "version": 1, "activation", "input_size", "layers": [{"neurons": [{"weights", "bias"}]}]}` |
| `.txt` | human-readable listing, every value preceded by a keyword; parseable with `fscanf` |
| `.h` | C constants: `<PREFIX>_INPUT_SIZE`, `<PREFIX>_LAYER_COUNT`, `<PREFIX>_LAYER_SIZES`, `<PREFIX>_L<n>_WEIGHTS`, `<PREFIX>_L<n>_BIASES` |

Floats are written with full round-trip precision, so a reload predicts bit-identically.

## Status

- [x] Part 1, steps 1–5 in Python: network, training, JSON/TXT export and import
- [x] Part 1, step 6 in Python: C header (constants only; the init function follows with the C structs of part 4)
- [x] Visualization page
- [ ] C# implementation
- [ ] Part 2 in both languages, results in the documentation
