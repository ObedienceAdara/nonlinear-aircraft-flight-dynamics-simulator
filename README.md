# Aircraft 6-DOF Flight Dynamics Simulator

A from-scratch, configurable nonlinear **six-degree-of-freedom aircraft flight-dynamics model (FDM)** in Python.

The repository now has one canonical implementation under `src/aircraft6dof/` and one project-level executable entry point: `main.py`.

## What is modeled

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
- verification tests;
- automatic CSV/JSON data export;
- automatic engineering visualization generation.

This is a **nonlinear flight-dynamics framework**, not a validated or flight-certified model of a particular aircraft. The default aerodynamic and propulsion parameters are generic demonstration values. High-fidelity aircraft results require aircraft-specific validated geometry, mass properties, aerodynamic databases, propulsion maps and environmental/actuator models.

## Run the complete project

From the repository root:

```bash
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Then run the complete simulation and reporting pipeline:

```bash
python main.py
```

A single run performs the nonlinear simulation, evaluates the derived flight-dynamics quantities, writes the complete time history, generates the engineering plots and creates a JSON summary.

## Generated output

Every run writes to `outputs/`:

```text
outputs/
├── simulation.csv
├── flight_history.png
├── trajectory_3d.png
├── attitude.png
├── position_ned.png
├── velocity.png
├── angular_rates.png
├── aerodynamic_angles.png
├── aerodynamic_forces.png
├── aerodynamic_moments.png
├── control_inputs.png
├── atmospheric_state.png
├── wind_and_gust.png
├── propulsion.png
├── dynamic_pressure.png
├── mach_number.png
├── flight_path.png
└── simulation_summary.json
```

### `simulation.csv`

The machine-readable time history contains position, altitude, body/NED velocity, attitude, quaternion state, angular rates, aerodynamic angles, dynamic pressure, Mach number, aerodynamic coefficients, aerodynamic forces and moments, propulsion, control inputs, wind, gust and atmospheric quantities.

### Engineering plots

The PNG plot pack provides separate views of:

- complete flight history;
- 3D trajectory;
- roll/pitch/yaw attitude;
- North/East/Down position;
- body and airspeed velocity;
- p/q/r angular rates;
- angle of attack and sideslip;
- aerodynamic forces and moments;
- control-surface commands and throttle;
- atmospheric temperature, pressure and density;
- steady wind and deterministic gust;
- propulsion thrust;
- dynamic pressure;
- Mach number;
- flight-path angle.

## Verify the implementation

```bash
pytest
```

The same core verification is run in GitHub Actions, followed by `python main.py` as an end-to-end execution check.

## Architecture

```text
project-level main.py
        │
        ├── aircraft definition
        ├── control schedule
        ├── atmosphere / wind / gust
        │
        ▼
   Simulator + RK4
        │
        ▼
 nonlinear 6-DOF equations
        │
        ├── aerodynamics
        ├── propulsion
        ├── gravity
        ├── quaternion attitude
        └── navigation
        │
        ▼
 SimulationHistory
        │
        ▼
 reporting.py
        ├── simulation.csv
        ├── simulation_summary.json
        └── engineering plots
```

## Repository layout

```text
.
├── main.py
├── requirements.txt
├── pyproject.toml
├── src/
│   └── aircraft6dof/
│       ├── aircraft.py
│       ├── actuators.py
│       ├── aero.py
│       ├── atmosphere.py
│       ├── constants.py
│       ├── equations.py
│       ├── frames.py
│       ├── geodesy.py
│       ├── gravity.py
│       ├── integrators.py
│       ├── mathutils.py
│       ├── propulsion.py
│       ├── reporting.py
│       ├── simulation.py
│       ├── state.py
│       └── wind.py
├── tests/
└── docs/
```

## Core equations

Translational:

```text
m (v_dot + omega × v) = F
```

Rotational:

```text
I omega_dot + omega × (I omega) = M
```

Quaternion:

```text
q_dot = 1/2 q ⊗ [0, p, q, r]
```

Aerodynamic dynamic pressure:

```text
qbar = 1/2 rho V²
```

Relative velocity:

```text
V_rel,N = V_aircraft,N - V_wind,N - V_gust,N
V_rel,B = C_NB V_rel,N
```

## Validation boundary

The project is intentionally explicit about model fidelity. The current canonical implementation uses a local flat-Earth NED rigid-body formulation with generic aerodynamic and propulsion parameters. It does not claim aircraft-specific validation, certification, full Earth-rotation navigation, exact MIL-F-8785C Dryden implementation, CFD-derived coefficient tables, or flight-test correlation.

The next fidelity layers are:

- full WGS-84 geodetic navigation and Earth rotation;
- validated Dryden/MIL-F-8785C turbulence;
- aircraft-specific aerodynamic lookup tables;
- engine/propeller performance maps;
- integrated actuator and sensor dynamics;
- landing gear and ground contact;
- trim and linearization tooling;
- stability/mode analysis;
- classical flight-control laws;
- hardware-in-the-loop interfaces;
- reinforcement-learning environment wrappers.

See `docs/EQUATIONS.md`, `docs/ARCHITECTURE.md` and `docs/VALIDATION.md`.
