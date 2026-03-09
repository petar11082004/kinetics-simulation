from pathlib import Path

import numpy as np

from kinetics import KineticsSystem, Reaction
from kinetics.io_utils import parse_reaction_line
from kinetics.tasks import oregonator_timecourse, protein_equilibrium_curve


def test_parse_reaction_line_with_stoichiometry() -> None:
    reaction = parse_reaction_line("2 X + Y -> 3 Z ; kf=4.2 ; label=test")
    assert reaction.reactants == {"X": 2, "Y": 1}
    assert reaction.products == {"Z": 3}
    assert reaction.kf0 == 4.2
    assert reaction.kb0 == 0.0
    assert reaction.label == "test"


def test_two_state_reversible_reaches_expected_equilibrium() -> None:
    reactions = [Reaction(reactants={"A": 1}, products={"B": 1}, kf0=2.0, kb0=1.0)]
    system = KineticsSystem(reactions)

    final_state, info = system.integrate_to_equilibrium(
        initial={"A": 1.0, "B": 0.0},
        dt=1.0e-3,
        tol=1.0e-12,
        max_steps=200_000,
    )

    assert info.converged
    total = final_state["A"] + final_state["B"]
    frac_b = final_state["B"] / total
    assert np.isclose(frac_b, 2.0 / 3.0, atol=2.0e-3)


def test_protein_folding_curve_has_expected_trend() -> None:
    reaction_file = Path("data/protein_reactions.txt")
    initial_file = Path("data/protein_initial_conditions.txt")
    urea_values = np.asarray([0.0, 4.0, 8.0])

    table, converged = protein_equilibrium_curve(
        reaction_file=reaction_file,
        initial_file=initial_file,
        urea_values=urea_values,
        dt=2.0e-6,
        tol=1.0e-6,
        max_steps=250_000,
    )

    assert all(converged)

    frac_d = table[:, 1]
    frac_n = table[:, 3]

    assert frac_d[0] < frac_d[-1]
    assert frac_n[0] > frac_n[-1]
    assert frac_n[0] > 0.9
    assert frac_d[-1] > 0.9


def test_oregonator_short_run_shapes_and_non_negative() -> None:
    times, history = oregonator_timecourse(
        reaction_file="data/oregonator_reactions.txt",
        initial_file="data/oregonator_initial_conditions.txt",
        total_time=0.2,
        sample_dt=0.02,
        method="BDF",
    )

    assert times.ndim == 1
    assert len(times) > 5

    for species in ("X", "Y", "Z"):
        assert species in history
        assert history[species].shape == times.shape
        assert np.all(history[species] >= 0.0)
