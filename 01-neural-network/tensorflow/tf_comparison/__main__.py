"""Command line: compare the own implementation with TensorFlow.

Run from the ``tensorflow`` folder::

    python -m tf_comparison
    python -m tf_comparison xor --epochs 5000
"""

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from tf_comparison.comparison import (
    CONVERGED_COST,
    EPOCHS,
    LEARNING_RATE,
    PROBLEMS,
    Comparison,
    compare,
)

COST_CHECKPOINTS = (0, 2000, 4000, 6000, 10000, 20000)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and print the comparison of the chosen problems.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments without the program name; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Exit code: 0 if the same-start runs agree to 1e-6.
    """
    args = _build_parser().parse_args(argv)
    names = list(PROBLEMS) if args.problem == "all" else [args.problem]

    agree = True
    for index, name in enumerate(names):
        if index > 0:
            print()
        comparison = compare(PROBLEMS[name], args.epochs)
        _print_comparison(comparison)
        agree &= comparison.max_difference < 1e-6
    return 0 if agree else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tf_comparison",
        description="Compare the own neural network with TensorFlow (INSY).",
    )
    parser.add_argument("problem", nargs="?", default="all", choices=["all", *PROBLEMS])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    return parser


def _print_comparison(comparison: Comparison) -> None:
    problem = comparison.problem
    runs = (comparison.own, comparison.same_start, comparison.keras_default)
    architecture = "-".join(str(size) for size in problem.layer_sizes)
    print(
        f"{problem.name} {architecture}, {comparison.epochs} epochs, "
        f"alpha {LEARNING_RATE}, cost J = 1/2 * sum((y_hat - y)^2)"
    )
    print()

    width = max(len(run.label) for run in runs)
    print(f"{'run':<{width}}  final cost  correct  cost < {CONVERGED_COST}   time")
    for run in runs:
        converged = run.converged_at()
        reached = f"epoch {converged}" if converged is not None else "never"
        correct = f"{run.correct(problem.targets)}/{len(problem.targets)}"
        print(
            f"{run.label:<{width}}  {run.costs[-1]:10.6f}  {correct:>7}  "
            f"{reached:<13}  {run.seconds:5.1f} s"
        )

    print()
    print(f"{'epoch':>6}  {'own':>10}  {'TF same':>10}  {'TF Keras':>10}")
    for epoch in (e for e in COST_CHECKPOINTS if e <= comparison.epochs):
        costs = (run.costs[epoch] for run in runs)
        print(f"{epoch:>6}" + "".join(f"  {cost:10.6f}" for cost in costs))

    if len(problem.inputs) <= 4:
        print()
        print(f"{'input':<8}  target  {'own':>7}  {'TF same':>7}  {'TF Keras':>8}")
        for index, (inputs, targets) in enumerate(
            zip(problem.inputs, problem.targets, strict=True)
        ):
            outputs = (run.outputs[index][0] for run in runs)
            print(
                f"{inputs!s:<8}  {targets[0]:>6}"
                + "".join(f"  {value:>7.4f}" for value in outputs)
            )

    print()
    errors = ", ".join(
        f"{label} {np.max(np.abs(run.outputs - np.asarray(problem.targets))):.3f}"
        for label, run in zip(("own", "TF same", "TF Keras"), runs, strict=True)
    )
    print(f"largest |output - target|: {errors}")
    print(
        "own vs TensorFlow with the same start: largest output difference "
        f"{comparison.max_difference:.1e}"
    )


if __name__ == "__main__":
    sys.exit(main())
