# Aircraft 6-DOF Flight Dynamics Simulator

A from-scratch, configurable nonlinear **six-degree-of-freedom aircraft flight-dynamics model (FDM)** in Python.

The original learning chapters remain in the repository for traceability, but the canonical simulator is now under `src/aircraft6dof/`.

## What is actually modeled

The canonical FDM includes:

- nonlinear rigid-body translational and rotational dynamics;
- quaternion attitude propagation;
- local North-East-Down navigation;
- standard-atmosphere temperature, pressure, density and speed of sound;
- latitude/altitude normal gravity;
- steady wind;
- deterministic one-minus-cosine gusts;
- reproducible stochastic turbulence / Dryden-style shaping;
- relative-air-velocity calculation;
- angle of attack, sideslip, dynamic pressure and Mach;
- nonlinear aerodynamic coefficient buildup;
- static and dynamic stability derivatives;
- aileron/elevator/rudder effects;
- propulsion/thrust and thrust moments;
- optional actuator dynamics with rate/position limits;
- RK4 time integration;
- unit/invariant tests;
- executable examples and plots.

This is a **real nonlinear flight-dynamics framework**, but not a validated or flight-certified model of a particular aircraft. The default aerodynamic and propulsion parameters are generic demonstration values. High-fidelity results require aircraft-specific validated data.

## Core equations

Translational:

```
m (v_dot + omega × v) = F
```

Rotational:

```
I omega_dot + omega × (I omega) = M
```

Quaternion:

```
q_dot = 1/2 q ⊗ [0, p, q, r]
```

Aerodynamic dynamic pressure:

```
qbar = 1/2 rho V^2
```

Relative velocity:

```
V_rel,N = V_aircraft,N - V_wind,N - V_gust,N
V_rel,B = C_NB V_rel,N
```

## Canonical architecture

```
controller / pilot commands
          ↓
      actuators
          ↓
   control surfaces
          ↓
 ┌────────┴─────────┐
 ↓                  ↓
aerodynamics     propulsion
 ↑                  ↑
relative wind    throttle
 ↑
atmosphere + wind + gust
          ↓
   forces + moments
          ↓
  6-DOF rigid body
          ↓
 quaternion + position
          ↓
      integration
          ↓
       next state
```

The canonical package is deliberately modular so aircraft-specific data can be substituted without rewriting the rigid-body equations.

## Repository layout

```
.
├── src/
│   └── aircraft6dof/          # canonical simulator
├── tests/                      # verification
├── examples/                   # runnable demonstrations
├── docs/
│   ├── ARCHITECTURE.md
│   └── EQUATIONS.md
├── chapter-01-6dof-equations/
├── chapter-02-euler-kinematics/
├── chapter-03-navigation-atmosphere-aerodynamics/
├── pyproject.toml
└── .github/workflows/
    └── core.yml
```

## Install

Python 3.10+:

```bash
python -m venv .venv
```

Activate it, then:

```bash
python -m pip install -e ".[dev]"
```

## Verify

Run the complete test suite:

```bash
pytest
```

The same checks run in GitHub Actions.

## Run the example

```bash
python examples/run_simulation.py
python examples/plot_simulation.py
```

Outputs are written to:

```
outputs/simulation.csv
outputs/flight_history.png
```

## Validation philosophy

The simulator follows an incremental verification strategy:

1. verify mathematical identities and frame transformations;
2. verify limiting cases such as zero rates, zero coefficients and zero wind;
3. verify numerical integration behavior;
4. verify environment models against standard reference values;
5. only then introduce aircraft-specific data and controllers.

The goal is to prevent a numerically stable but physically incorrect simulator.

## Fidelity roadmap

The next engineering layers are not additional tutorial copies. They are model-fidelity upgrades:

- full WGS-84 geodetic navigation and Earth rotation;
- fully validated Dryden/MIL-F-8785C turbulence;
- aircraft-specific aerodynamic databases / lookup tables;
- engine/propeller performance maps;
- actuator and sensor dynamics;
- landing gear and ground contact;
- trim and linearization tooling;
- stability/mode analysis;
- classical flight-control laws;
- hardware-in-the-loop interfaces;
- reinforcement-learning environment wrappers.

## References

The implementation is informed by established flight-dynamics formulations rather than treating the YouTube playlist as the authoritative source.

- JSBSim's current reference manual documents a nonlinear 6-DOF FDM, equations of motion, configurable aerodynamics/propulsion, and Earth/atmosphere modeling.
- NASA technical material documents six-DOF aircraft simulation architectures and multiple Dryden turbulence implementations.

See [docs/EQUATIONS.md](docs/EQUATIONS.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Status

**Canonical simulator: active development**

Chapters 01–03 are retained as the learning trail; `src/aircraft6dof` is the source of truth for future development.
