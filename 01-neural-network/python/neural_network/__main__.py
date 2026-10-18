"""Command line: train the assignment networks, export them, build the page.

Run from the ``python`` folder::

    python -m neural_network xor
    python -m neural_network seven-segment --export ../models/python
    python -m neural_network visualize
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neural_network.datasets import (
    SEVEN_SEGMENT_INPUTS,
    SEVEN_SEGMENT_TARGETS,
    XOR_INPUTS,
    XOR_TARGETS,
)
from neural_network.network import NeuralNetwork

SEGMENT_NAMES = "abcdefg"


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the chosen command.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments without the program name; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Exit code: 0 if every sample is predicted correctly.
    """
    args = _build_parser().parse_args(argv)
    return args.handler(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neural_network",
        description="Neural network from scratch (INSY).",
    )
    commands = parser.add_subparsers(required=True, metavar="command")

    xor = commands.add_parser("xor", help="train the XOR gate (2-2-1)")
    _add_training_arguments(xor, seed=0)
    xor.add_argument("--hidden", type=int, default=2, help="hidden neurons")
    xor.set_defaults(handler=_run_xor)

    seven = commands.add_parser(
        "seven-segment", help="train the 7-segment decoder (4-16-7)"
    )
    _add_training_arguments(seven, seed=42)
    seven.add_argument("--hidden", type=int, default=16, help="hidden neurons")
    seven.set_defaults(handler=_run_seven_segment)

    visualize = commands.add_parser("visualize", help="build the HTML page")
    visualize.add_argument(
        "--output",
        type=Path,
        default=Path("../visualization/index.html"),
        help="target file (default: %(default)s)",
    )
    visualize.set_defaults(handler=_run_visualize)

    return parser


def _add_training_arguments(parser: argparse.ArgumentParser, seed: int) -> None:
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate alpha")
    parser.add_argument("--seed", type=int, default=seed)
    parser.add_argument(
        "--export",
        type=Path,
        metavar="DIR",
        help="write .json, .txt and .h to DIR and verify the reload",
    )


def _run_xor(args: argparse.Namespace) -> int:
    network = _train("XOR", [2, args.hidden, 1], XOR_INPUTS, XOR_TARGETS, args)

    print()
    print("input    target  output   rounded")
    all_correct = True
    for inputs, targets in zip(XOR_INPUTS, XOR_TARGETS, strict=True):
        output = network.predict(inputs)[0]
        correct = round(output) == targets[0]
        all_correct &= correct
        print(
            f"{inputs!s:8} {targets[0]:<7} {output:.4f}   {round(output)}"
            f"  {'ok' if correct else 'WRONG'}"
        )

    _export_if_requested(network, args, "xor", XOR_INPUTS)
    return 0 if all_correct else 1


def _run_seven_segment(args: argparse.Namespace) -> int:
    network = _train(
        "7-segment",
        [4, args.hidden, 7],
        SEVEN_SEGMENT_INPUTS,
        SEVEN_SEGMENT_TARGETS,
        args,
    )

    outputs = [network.predict(x) for x in SEVEN_SEGMENT_INPUTS]
    print()
    _print_digits(outputs)
    print()

    correct_digits = 0
    for digit, (output, targets) in enumerate(
        zip(outputs, SEVEN_SEGMENT_TARGETS, strict=True)
    ):
        wrong = [
            SEGMENT_NAMES[i]
            for i, (value, target) in enumerate(zip(output, targets, strict=True))
            if round(value) != target
        ]
        correct_digits += not wrong
        values = " ".join(f"{value:.2f}" for value in output)
        status = "ok" if not wrong else f"WRONG segments {','.join(wrong)}"
        print(f"{digit}: {values}  {status}")
    print(f"\n{correct_digits}/10 digits correct")

    _export_if_requested(network, args, "seven_segment", SEVEN_SEGMENT_INPUTS)
    return 0 if correct_digits == len(SEVEN_SEGMENT_INPUTS) else 1


def _run_visualize(args: argparse.Namespace) -> int:
    # Imported here: only this command needs the page builder
    from neural_network.visualization import build_page

    build_page(args.output, progress=print)
    print(f"written {args.output.resolve()}")
    return 0


def _train(
    name: str,
    layer_sizes: list[int],
    inputs: list[list[int]],
    targets: list[list[int]],
    args: argparse.Namespace,
) -> NeuralNetwork:
    network = NeuralNetwork(layer_sizes, args.lr, args.seed)
    architecture = "-".join(str(size) for size in layer_sizes)
    print(
        f"{name} {architecture}, {args.epochs} epochs, "
        f"alpha {args.lr}, seed {args.seed}"
    )
    print(f"epoch {0:>6}  cost {network.mean_cost(inputs, targets):.6f}")

    report_every = max(1, args.epochs // 10)

    def report(epoch: int, cost: float) -> None:
        if epoch % report_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:>6}  cost {cost:.6f}")

    network.fit(inputs, targets, args.epochs, on_epoch=report)
    return network


def _export_if_requested(
    network: NeuralNetwork,
    args: argparse.Namespace,
    name: str,
    inputs: list[list[int]],
) -> None:
    if args.export is None:
        return

    directory: Path = args.export
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{name}.json"
    txt_path = directory / f"{name}.txt"
    header_path = directory / f"{name}.h"

    network.dump_json(json_path)
    network.dump_txt(txt_path)
    network.dump_initfile(header_path, name)

    print()
    for path, loaded in (
        (json_path, NeuralNetwork.load(json_path)),
        (txt_path, NeuralNetwork.load_txt(txt_path)),
    ):
        difference = max(
            float(np.max(np.abs(network.predict(x) - loaded.predict(x))))
            for x in inputs
        )
        print(f"reloaded {path}: max difference {difference:.1e}")
    print(f"written  {header_path}")


def _print_digits(outputs: list[np.ndarray]) -> None:
    rows = ["", "", ""]
    for output in outputs:
        a, b, c, d, e, f, g = (round(value) == 1 for value in output)
        rows[0] += " " + ("_" if a else " ") + "   "
        rows[1] += (
            ("|" if f else " ") + ("_" if g else " ") + ("|" if b else " ") + "  "
        )
        rows[2] += (
            ("|" if e else " ") + ("_" if d else " ") + ("|" if c else " ") + "  "
        )
    for row in rows:
        print(row.rstrip())
    print("".join(f" {digit}   " for digit in range(10)).rstrip())


if __name__ == "__main__":
    sys.exit(main())
