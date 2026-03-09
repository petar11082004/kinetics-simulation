"""Task-specific helpers for Exercise 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from .io_utils import load_initial_conditions, load_reactions
from .model import KineticsSystem


def protein_equilibrium_curve(
    reaction_file: str | Path,
    initial_file: str | Path,
    urea_values: np.ndarray,
    dt: float = 2.0e-6,
    tol: float = 1.0e-6,
    max_steps: int = 400_000,
) -> tuple[np.ndarray, list[bool]]:
    """Simulate protein fractions at equilibrium for each urea concentration."""
    system = KineticsSystem(load_reactions(reaction_file))
    initial = load_initial_conditions(initial_file)

    table = np.zeros((len(urea_values), 4), dtype=float)
    converged_flags: list[bool] = []

    for i, urea in enumerate(urea_values):
        current = dict(initial)
        info = None

        # Retry a few blocks if one block is not enough to satisfy strict tolerance.
        for _ in range(3):
            current, info = system.integrate_to_equilibrium(
                initial=current,
                dt=dt,
                tol=tol,
                max_steps=max_steps,
                context={"urea": float(urea)},
            )
            if info.converged:
                break

        final_state = current
        assert info is not None

        total = sum(final_state.get(species, 0.0) for species in ("D", "I", "N"))
        if total <= 0.0:
            fractions = (0.0, 0.0, 0.0)
        else:
            fractions = (
                final_state.get("D", 0.0) / total,
                final_state.get("I", 0.0) / total,
                final_state.get("N", 0.0) / total,
            )

        table[i] = (urea, *fractions)
        converged_flags.append(info.converged)

    return table, converged_flags


def oregonator_timecourse(
    reaction_file: str | Path,
    initial_file: str | Path,
    total_time: float = 90.0,
    sample_dt: float = 0.05,
    method: str = "BDF",
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Simulate Oregonator concentration traces with scipy.integrate.solve_ivp."""
    system = KineticsSystem(load_reactions(reaction_file))
    initial = load_initial_conditions(initial_file)

    species = list(system.species)
    y0 = np.asarray([initial.get(sp, 0.0) for sp in species], dtype=float)

    t_eval = np.arange(0.0, total_time + sample_dt, sample_dt)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        state = {sp: float(max(val, 0.0)) for sp, val in zip(species, y)}
        grad = system.derivative(state)
        return np.asarray([grad[sp] for sp in species], dtype=float)

    solution = solve_ivp(
        rhs,
        t_span=(0.0, total_time),
        y0=y0,
        method=method,
        t_eval=t_eval,
        atol=1.0e-16,
        rtol=1.0e-8,
    )

    if not solution.success:
        raise RuntimeError(f"Oregonator integration failed: {solution.message}")

    history = {
        sp: np.clip(solution.y[i], a_min=0.0, a_max=None)
        for i, sp in enumerate(species)
    }
    return solution.t, history
