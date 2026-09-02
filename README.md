# Aircraft 6-DOF Flight Dynamics Simulator

A from-scratch, configurable nonlinear **six-degree-of-freedom aircraft flight-dynamics model (FDM)** in Python.

The repository has one canonical implementation under `src/aircraft6dof/` and one project-level executable entry point: `main.py`.

## Run the complete project

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python main.py
```

A single `main.py` run executes the nonlinear 6-DOF simulation and produces the complete machine-readable data set, engineering visualization pack and JSON summary under `outputs/`.

## Generated output

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

`simulation.csv` contains the full time history of position, altitude, body/NED velocity, attitude, quaternion state, angular rates, aerodynamic angles, dynamic pressure, Mach number, aerodynamic coefficients, aerodynamic forces and moments, propulsion, control inputs, wind, gust and atmosphere.

The PNG pack provides dedicated engineering views for trajectory, attitude, navigation, velocity, angular rates, aerodynamic state, forces, moments, controls, atmosphere, wind/gust, propulsion, dynamic pressure, Mach and flight-path angle.

## Architecture

```text
main.py
  │
  ├── aircraft definition
  ├── control schedule
  └── atmosphere / wind / gust
  │
  ▼
Simulator + RK4
  │
  ▼
nonlinear 6-DOF equations
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

## Physics model

The canonical FDM includes nonlinear rigid-body translational and rotational dynamics, quaternion attitude propagation, local NED navigation, standard-atmosphere properties, normal gravity, steady wind, deterministic one-minus-cosine gusts, Dryden-style stochastic shaping, relative-air-velocity calculation, angle of attack, sideslip, dynamic pressure, Mach number, nonlinear aerodynamic coefficient buildup, stability derivatives, control-surface effects, propulsion/thrust and thrust moments, optional actuator dynamics and RK4 integration.

Core equations:

```text
m (v_dot + omega × v) = F
I omega_dot + omega × (I omega) = M
q_dot = 1/2 q ⊗ [0, p, q, r]
qbar = 1/2 rho V²
V_rel,N = V_aircraft,N - V_wind,N - V_gust,N
```

## Verification

```bash
pytest
```

GitHub Actions runs the verification suite and then executes `python main.py` as an end-to-end project check.

## Fidelity boundary

This is a **nonlinear flight-dynamics framework**, not a validated or flight-certified model of a particular aircraft. The demonstration uses generic aerodynamic and propulsion parameters. Aircraft-specific high-fidelity results require validated geometry, mass properties, aerodynamic databases, propulsion maps, actuator/sensor models and environmental parameters.

The current canonical equations use a local flat-Earth NED formulation. WGS-84 geodetic navigation and Earth-rotation corrections, exact validated Dryden/MIL-F-8785C turbulence, aircraft-specific aerodynamic tables, engine maps, ground contact, trim/linearization, flight-control laws, HIL interfaces and RL wrappers are future fidelity layers.

See `docs/EQUATIONS.md`, `docs/ARCHITECTURE.md` and `docs/VALIDATION.md`.
