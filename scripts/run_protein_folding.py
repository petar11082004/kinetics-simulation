#!/usr/bin/env python3
"""Run Exercise 3 Task 1: protein folding equilibrium fractions vs urea."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinetics.tasks import protein_equilibrium_curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", default="data/protein_reactions.txt")
    parser.add_argument("--initial", default="data/protein_initial_conditions.txt")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--dt", type=float, default=2.0e-6)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-steps", type=int, default=400_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    urea_values = np.linspace(0.0, 8.0, args.points)
    table, converged_flags = protein_equilibrium_curve(
        reaction_file=args.reactions,
        initial_file=args.initial,
        urea_values=urea_values,
        dt=args.dt,
        tol=args.tol,
        max_steps=args.max_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = output_dir / "protein_equilibrium.dat"
    np.savetxt(
        data_path,
        table,
        header="urea_M frac_D frac_I frac_N",
        comments="# ",
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(table[:, 0], table[:, 1], "o-", lw=1.3, ms=3, label="D")
    plt.plot(table[:, 0], table[:, 2], "o-", lw=1.3, ms=3, label="I")
    plt.plot(table[:, 0], table[:, 3], "o-", lw=1.3, ms=3, label="N")
    plt.xlabel("[Urea] / M")
    plt.ylabel("Fraction of species")
    plt.title("Equilibrium fractions of protein states against urea concentration")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    fig_path = output_dir / "protein_equilibrium.png"
    plt.savefig(fig_path, dpi=160)

    converged_count = sum(converged_flags)
    print(f"Saved data: {data_path}")
    print(f"Saved figure: {fig_path}")
    print(f"Converged points: {converged_count}/{len(converged_flags)}")


if __name__ == "__main__":
    main()
