# Oregonator spatial extension (Task 2b)

A simple way to add spatial resolution is a 1D grid of cells.

1. Data model
- Keep one `KineticsSystem` object (same local chemistry everywhere).
- Store concentrations in a 2D array: `conc[cell_index, species_index]`.

2. One timestep
- Reaction step: for every cell, compute local `dC/dt` from kinetics and apply an Euler update.
- Diffusion step: for every species with diffusion coefficient `D`, apply
  `dC/dt = D * (C[i-1] - 2*C[i] + C[i+1]) / dx^2`.

3. Boundary conditions
- Start with no-flux boundaries: left and right ghost values equal edge cell values.

4. Solver loop
- Alternate reaction and diffusion updates each `dt` (operator splitting).
- Save snapshots every `n` steps for plotting space-time patterns.

5. Plotting
- For one coloured species, plot `concentration(x, t)` as a heatmap.
- This directly shows travelling/standing wave patterns.

This design is easy to test because with `D = 0` it should reduce exactly to the well-mixed model.
