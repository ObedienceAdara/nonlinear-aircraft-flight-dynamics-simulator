# Chapter 02 — Euler Attitude Kinematics

Source lesson: [Aircraft Euler Kinematics (Attitude) Simulation in Python — Section 1.3](https://youtu.be/ERVrlzttLg4)

This chapter extends Chapter 01 with attitude propagation. It implements the relationship between body angular rates `p, q, r` and the time derivatives of the aircraft Euler angles `phi, theta, psi`.

## Core equation

For the conventional aircraft 3-2-1 (yaw-pitch-roll) Euler-angle convention:

```
phi_dot   = p + tan(theta) * (q sin(phi) + r cos(phi))
theta_dot = q cos(phi) - r sin(phi)
psi_dot   = (q sin(phi) + r cos(phi)) / cos(theta)
```

or:

```
eta_dot = T(phi, theta) @ [p, q, r]
```

with:

```
T = [[1, sin(phi)tan(theta),  cos(phi)tan(theta)],
     [0, cos(phi),            -sin(phi)],
     [0, sin(phi)/cos(theta),  cos(phi)/cos(theta)]]
```

Angles are in radians.

## Why this exists

Body rates are rotations resolved about the instantaneous body axes. Euler-angle rates describe the rates of the chosen attitude coordinates. They are therefore not generally equal:

```
phi_dot != p
theta_dot != q
psi_dot != r
```

except in special cases.

## Euler-angle singularity

The transformation contains `tan(theta)` and `1/cos(theta)`. It is singular at:

```
theta = ±pi/2
```

The implementation explicitly detects proximity to this singularity instead of silently returning huge numerical values.

Euler angles remain useful for normal aircraft flight because their interpretation is intuitive, but later versions of this simulator can add quaternion propagation when large-attitude or globally nonsingular motion is required.

## Coordinate transformations

This chapter also adds direction-cosine matrices (DCMs) for the same 3-2-1 convention:

```
v_navigation = C_body_to_navigation @ v_body
v_body       = C_navigation_to_body @ v_navigation
```

The matrices are transpose/inverse pairs for a proper orthonormal rotation matrix.

The repository uses a North-East-Down (NED) navigation convention:

```
x_N -> North
y_E -> East
z_D -> Down
```

## Demo

`main.py` runs two simulations:

1. a pure constant body-roll-rate case;
2. a coupled 3-axis body-rate case.

Both propagate Euler attitude with RK4. The output CSV contains body rates and Euler angles, and `plot_results.py` visualizes their evolution.

This is still a kinematics demonstration, not a complete aircraft model. Chapter 01's force/moment dynamics and later aerodynamic/navigation chapters will be integrated progressively.

## Run

From `chapter-02-euler-kinematics/`:

```bash
pip install -r requirements.txt
python main.py
python plot_results.py
pytest -q
```

## What to study

Read:

1. `aircraft6dof/kinematics.py` — derivation translated directly into code.
2. `aircraft6dof/frames.py` — 3-2-1 DCM construction and frame conversion.
3. `main.py` — how the kinematics becomes a time simulation.
4. `tests/test_kinematics.py` — numerical identities and singularity behavior.

## Relation to Chapter 01

Chapter 01 computes:

```
[u_dot, v_dot, w_dot, p_dot, q_dot, r_dot]
```

Chapter 02 computes:

```
[phi_dot, theta_dot, psi_dot]
```

Together they begin forming the coupled rigid-body aircraft state derivative.

## Important source note

The implementation is an independent reconstruction of the lesson's concepts. It does not reproduce proprietary/source-locked lesson files. The video author's later errata are treated as corrections where applicable.
