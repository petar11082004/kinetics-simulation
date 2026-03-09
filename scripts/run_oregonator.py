#!/usr/bin/env python3
"""Run Exercise 3 Task 2(a): Oregonator dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinetics.tasks import oregonator_timecourse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reactions", default="data/oregonator_reactions.txt")
    parser.add_argument("--initial", default="data/oregonator_initial_conditions.txt")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--time", type=float, default=90.0)
    parser.add_argument("--sample-dt", type=float, default=0.05)
    parser.add_argument("--method", default="BDF", choices=["BDF", "Radau", "LSODA"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    times, history = oregonator_timecourse(
        reaction_file=args.reactions,
        initial_file=args.initial,
        total_time=args.time,
        sample_dt=args.sample_dt,
        method=args.method,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x = history.get("X")
    y = history.get("Y")
    z = history.get("Z")
    if x is None or y is None or z is None:
        raise RuntimeError("Expected X, Y, Z in Oregonator output.")

    data_matrix = np.column_stack([times, x, y, z])
    data_path = output_dir / "oregonator_timeseries.dat"
    np.savetxt(
        data_path,
        data_matrix,
        header="time_s X_M Y_M Z_M",
        comments="# ",
    )

    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(times, x, label="X")
    plt.plot(times, y, label="Y")
    plt.plot(times, z, label="Z")
    plt.yscale("log")
    plt.xlabel("t / s")
    plt.ylabel("Concentration / M")
    plt.title("Oregonator concentrations over time")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    fig_path = output_dir / "oregonator_timeseries.png"
    plt.savefig(fig_path, dpi=160)

    print(f"Saved data: {data_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
