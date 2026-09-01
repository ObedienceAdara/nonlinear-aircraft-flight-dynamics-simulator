# Canonical Simulator Architecture

The repository now has two roles:

1. **chapter-01..03/** — the historical, incremental implementations from the learning sequence.
2. **src/aircraft6dof/** — the canonical simulator. New physics belongs here.

## Physics pipeline

```
Pilot / controller commands
        |
        v
 actuator dynamics (optional)
        |
        v
  control surface states
        |
        +-------------------------------+
        |                               |
        v                               v
 atmosphere + wind                propulsion
        |                               |
        v                               |
 relative wind                          |
        |                               |
        v                               |
 alpha, beta, V, qbar, Mach             |
        |                               |
        v                               |
 nonlinear aerodynamic model            |
        |                               |
        +---------------+---------------+
                        v
                 forces + moments
                        |
             +----------+-----------+
             |                      |
             v                      v
        gravity model        6-DOF rigid body
                                   |
                            body velocity
                            angular rates
                                   |
                     +-------------+-------------+
                     |                           |
                     v                           v
               quaternion attitude         NED position
                     |
                     v
               frame transforms
                     |
                     +------ feedback ------+
```

## Canonical state

The state is represented by:

```
N, E, D
u, v, w
p, q, r
q0, q1, q2, q3
```

That is 13 stored numbers, but only 12 physical degrees of freedom because the quaternion has a unit-norm constraint.

Quaternions are used in the canonical simulator to avoid the Euler-angle singularity that exists at pitch = ±90°.

Euler angles are still calculated for human-readable output.

## Force balance

```
m(v_dot + omega × v) = F_total
```

with

```
F_total = F_aero + F_prop + m g_body
```

## Moment balance

```
I omega_dot + omega × (I omega) = M_total
```

with

```
M_total = M_aero + M_prop
```

## Attitude

The body-to-NED quaternion obeys:

```
q_dot = 1/2 q ⊗ [0, p, q, r]
```

for the local flat-Earth convention used by the core.

## Aerodynamic model

The default generic aircraft uses coefficient buildup with:

- lift curve slope;
- elevator effectiveness;
- pitch-rate damping;
- quadratic induced/profile-drag approximation;
- side-force/sideslip terms;
- roll/yaw static and rate derivatives;
- aileron/rudder effects.

The coefficients are **generic demonstration parameters**, not the measured characteristics of a named aircraft.

For a real aircraft model, replace them with validated wind-tunnel, flight-test, DATCOM/CFD, or manufacturer data.

## Environment

The core includes:

- standard-atmosphere temperature/pressure/density/speed of sound;
- latitude/altitude normal-gravity approximation;
- steady NED wind;
- deterministic one-minus-cosine gust;
- reproducible stochastic turbulence process.

The stochastic turbulence class is deliberately labelled a **Dryden-style** shaping model. It should not be represented as a certification-grade MIL-F-8785C implementation without further validation against the governing spectral equations and verification data.

## Fidelity boundary

This is a real nonlinear 6-DOF **flight-dynamics framework**, not a flight-certified or aircraft-validated model.

High-fidelity results require aircraft-specific:

- geometry;
- mass properties;
- aerodynamic coefficient data;
- propulsive maps;
- actuator characteristics;
- sensor models;
- engine dynamics;
- landing-gear/contact models;
- validated turbulence parameters;
- appropriate Earth/atmosphere model for the mission.

The software architecture is designed so those models can be inserted without rewriting the rigid-body integrator.
