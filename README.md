# Exercise 3 - Kinetics Simulation

This repository contains a submit-ready solution for **Exercise 3** from the course guide:

1. Protein folding equilibrium vs urea concentration.
2. Oregonator time evolution with oscillatory dynamics.
3. A design sketch for spatially resolved simulation (Task 2b).

## Project structure

- `src/kinetics/model.py`: core reaction and integrator classes.
- `src/kinetics/io_utils.py`: parsers for reaction and initial-condition files.
- `src/kinetics/tasks.py`: task-level helper functions.
- `scripts/run_protein_folding.py`: generates protein equilibrium data and figure.
- `scripts/run_oregonator.py`: generates Oregonator time-series data and figure.
- `data/*.txt`: reaction definitions and initial conditions.
- `docs/spatial_extension_design.md`: short design for Task 2(b).
- `tests/test_kinetics.py`: basic correctness tests.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run tasks

Protein folding (Task 1):

```bash
python scripts/run_protein_folding.py
```

Outputs:
- `output/protein_equilibrium.dat`
- `output/protein_equilibrium.png`

Oregonator (Task 2a):

```bash
python scripts/run_oregonator.py --time 90 --sample-dt 0.05 --method BDF
```

Outputs:
- `output/oregonator_timeseries.dat`
- `output/oregonator_timeseries.png`

## Run tests

```bash
pytest -q
```

## Notes on the input format

Reaction file line format:

```text
reactants -> products ; kf=<value> ; kb=<value> ; mf=<value> ; mb=<value>
```

Examples:

```text
A + Y -> X + P ; kf=1.34
D <-> I ; kf=2.6e4 ; kb=6.0e-2 ; mf=-1.68 ; mb=0.95
```

- `mf` and `mb` apply urea dependence as
  - `kf(urea) = kf * exp(mf * [urea])`
  - `kb(urea) = kb * exp(mb * [urea])`
- If omitted, `kb`, `mf`, and `mb` default to `0`.
