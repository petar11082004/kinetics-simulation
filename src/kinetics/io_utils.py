"""File parsing helpers for reactions and initial concentrations."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import re

import numpy as np

from .model import Reaction, Stoichiometry


_SPECIES_RE = re.compile(r"^(?:(\d+)\s+)?([A-Za-z][A-Za-z0-9_]*)$")


def parse_species_block(block: str) -> Stoichiometry:
    """Parse one side of a reaction, e.g. '2 X + Y'."""
    cleaned = block.strip()
    if cleaned in {"", "0"}:
        return {}

    stoich: Stoichiometry = {}
    for token in cleaned.split("+"):
        token = token.strip()
        match = _SPECIES_RE.match(token)
        if not match:
            raise ValueError(f"Invalid species token: '{token}'")
        coeff_text, species = match.groups()
        coeff = int(coeff_text) if coeff_text is not None else 1
        stoich[species] = stoich.get(species, 0) + coeff

    return stoich


def parse_reaction_line(line: str) -> Reaction:
    """Parse one reaction line.

    Supported format:
    D <-> I ; kf=2.6e4 ; kb=6.0e-2 ; mf=-1.68 ; mb=0.95 ; label=R15
    """
    segments = [segment.strip() for segment in line.split(";") if segment.strip()]
    if not segments:
        raise ValueError("Empty reaction line.")

    equation = segments[0]
    if "<->" in equation:
        lhs, rhs = [part.strip() for part in equation.split("<->", 1)]
    elif "->" in equation:
        lhs, rhs = [part.strip() for part in equation.split("->", 1)]
    else:
        raise ValueError(f"Reaction line must contain '->' or '<->': {line}")

    params = {}
    for segment in segments[1:]:
        if "=" not in segment:
            raise ValueError(f"Expected key=value in segment: '{segment}'")
        key, value = [piece.strip() for piece in segment.split("=", 1)]
        params[key.lower()] = value

    kf = float(params.get("kf", 0.0))
    kb = float(params.get("kb", 0.0))
    mf = float(params.get("mf", 0.0))
    mb = float(params.get("mb", 0.0))
    label = params.get("label", "")

    return Reaction(
        reactants=parse_species_block(lhs),
        products=parse_species_block(rhs),
        kf0=kf,
        kb0=kb,
        m_forward=mf,
        m_backward=mb,
        label=label,
    )


def load_reactions(path: str | Path) -> list[Reaction]:
    """Load reactions from a text file."""
    file_path = Path(path)
    reactions: list[Reaction] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        reactions.append(parse_reaction_line(line))

    if not reactions:
        raise ValueError(f"No reactions found in file: {file_path}")
    return reactions


def load_initial_conditions(path: str | Path) -> dict[str, float]:
    """Load two-column species/concentration text file using numpy.loadtxt."""
    file_path = Path(path)
    rows = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)

    if not rows:
        raise ValueError(f"No initial concentrations found in file: {file_path}")

    data = np.loadtxt(StringIO("\n".join(rows)), dtype=str)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 2:
        raise ValueError(f"Initial condition file must have 2 columns: {file_path}")

    return {row[0]: float(row[1]) for row in data}