"""Core data structures and numerical integration for reaction kinetics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import numpy as np


ConcentrationMap = Dict[str, float]
Stoichiometry = Dict[str, int]


@dataclass
class Reaction:
    """One elementary reaction with optional urea-dependent rates."""

    reactants: Stoichiometry
    products: Stoichiometry
    kf0: float
    kb0: float = 0.0
    m_forward: float = 0.0
    m_backward: float = 0.0
    label: str = ""

    def rate_constants(self, context: ConcentrationMap | None = None) -> tuple[float, float]:
        """Return forward and backward rate constants for current context."""
        ctx = context or {}
        urea = float(ctx.get("urea", 0.0))
        kf = self.kf0 * math.exp(self.m_forward * urea)
        kb = self.kb0 * math.exp(self.m_backward * urea) if self.kb0 > 0.0 else 0.0
        return kf, kb


@dataclass
class IntegrationInfo:
    """Metadata from an integration run."""

    steps: int
    time: float
    converged: bool


class KineticsSystem:
    """A set of elementary reactions acting on a shared concentration state."""

    def __init__(self, reactions: list[Reaction]):
        if not reactions:
            raise ValueError("KineticsSystem needs at least one reaction.")

        self.reactions = reactions
        species: set[str] = set()
        for reaction in reactions:
            species.update(reaction.reactants)
            species.update(reaction.products)
        self.species = sorted(species)

    def _rate_term(self, concentrations: ConcentrationMap, stoich: Stoichiometry) -> float:
        value = 1.0
        for species, power in stoich.items():
            value *= concentrations.get(species, 0.0) ** power
        return value

    def derivative(
        self,
        concentrations: ConcentrationMap,
        context: ConcentrationMap | None = None,
    ) -> ConcentrationMap:
        """Compute dC/dt for every species at the current state."""
        grads = {species: 0.0 for species in self.species}

        for reaction in self.reactions:
            kf, kb = reaction.rate_constants(context)
            forward = kf * self._rate_term(concentrations, reaction.reactants)
            backward = kb * self._rate_term(concentrations, reaction.products)
            net = forward - backward

            for species, coeff in reaction.reactants.items():
                grads[species] -= coeff * net
            for species, coeff in reaction.products.items():
                grads[species] += coeff * net

        return grads

    def step(
        self,
        concentrations: ConcentrationMap,
        dt: float,
        context: ConcentrationMap | None = None,
        clip_negative: bool = True,
    ) -> ConcentrationMap:
        """Take one explicit Euler step."""
        grads = self.derivative(concentrations, context)
        new_state: ConcentrationMap = {}

        for species in self.species:
            value = concentrations.get(species, 0.0) + dt * grads[species]
            if clip_negative and value < 0.0:
                value = 0.0
            new_state[species] = value

        return new_state

    def integrate(
        self,
        initial: ConcentrationMap,
        dt: float,
        n_steps: int,
        context: ConcentrationMap | None = None,
        sample_every: int = 1,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Integrate for a fixed number of steps and return sampled trajectory."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be at least 1.")
        if sample_every < 1:
            raise ValueError("sample_every must be at least 1.")

        concentrations = {species: float(initial.get(species, 0.0)) for species in self.species}

        times = [0.0]
        history: dict[str, list[float]] = {
            species: [concentrations[species]] for species in self.species
        }

        for step_index in range(1, n_steps + 1):
            concentrations = self.step(concentrations, dt, context=context)

            if step_index % sample_every == 0 or step_index == n_steps:
                times.append(step_index * dt)
                for species in self.species:
                    history[species].append(concentrations[species])

        time_array = np.asarray(times, dtype=float)
        history_arrays = {
            species: np.asarray(values, dtype=float) for species, values in history.items()
        }
        return time_array, history_arrays

    def integrate_to_equilibrium(
        self,
        initial: ConcentrationMap,
        dt: float,
        tol: float,
        max_steps: int,
        context: ConcentrationMap | None = None,
        min_stable_steps: int = 50,
    ) -> tuple[ConcentrationMap, IntegrationInfo]:
        """Integrate until concentrations stop changing within tolerance."""
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if tol <= 0.0:
            raise ValueError("tol must be positive.")
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        concentrations = {species: float(initial.get(species, 0.0)) for species in self.species}

        stable_count = 0
        converged = False

        for step_index in range(1, max_steps + 1):
            new_state = self.step(concentrations, dt, context=context)
            max_delta = max(
                abs(new_state[species] - concentrations[species]) for species in self.species
            )

            concentrations = new_state
            if max_delta < tol:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= min_stable_steps:
                converged = True
                break

        info = IntegrationInfo(steps=step_index, time=step_index * dt, converged=converged)
        return concentrations, info
