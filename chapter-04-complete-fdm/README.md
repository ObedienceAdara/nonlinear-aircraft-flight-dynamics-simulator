# Canonical 6-DOF Flight-Dynamics Model

This directory is the conceptual entry point for the complete simulator. The source of truth is now the package in `src/aircraft6dof/`.

## Included now

- 6-DOF nonlinear rigid-body equations;
- quaternion attitude;
- local NED navigation;
- standard atmosphere;
- normal gravity;
- steady wind;
- deterministic one-minus-cosine gust;
- stochastic correlated turbulence / Dryden-style process;
- relative wind and air-data calculation;
- nonlinear coefficient-buildup aerodynamics with stability derivatives;
- thrust and thrust moment;
- actuator primitive;
- RK4 integration;
- tests and reproducible examples;
- WGS-84 geodesy helpers for future navigation fidelity.

## Run

From repository root:

```bash
python -m pip install -e ".[dev]"
pytest
python examples/run_simulation.py
python examples/plot_simulation.py
```

## Critical fidelity statement

There is no universal "real aircraft" model without aircraft-specific measured or validated data. This repository supplies the governing mechanics and a generic configurable aerodynamic/propulsive model. It is suitable for engineering study and controller/RL experimentation after validation, but it is not a flight-certified FDM.

The strongest next validation step is to replace the generic coefficient set with a published aircraft data set and compare trim, static stability, dynamic modes, and nonlinear responses against independent references.
