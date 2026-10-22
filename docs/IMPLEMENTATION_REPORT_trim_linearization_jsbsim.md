# Implementation report: trim, linearization and flight-dynamics modes

> **Scope:** This report covers the trim, local-linearization and mode-analysis milestone. The JSBSim comparison layer is not included in this commit; it remains a separate follow-on milestone.

## Architecture introduced

```text
nonlinear 6-DOF model
        |
        v
    trim result
        |
        v
12-state Euler operating point
        |
        v
central-difference Jacobians
       A, B
        |
        v
linear perturbation model
        |
        v
flight-dynamics mode classification
```

The new layer reuses the canonical nonlinear derivative instead of building a second aerodynamic or rigid-body model.

## Implementation

- `src/aircraft6dof/linearize.py` converts between the quaternion-backed `AircraftState` and a 12-state Euler vector, builds the state Jacobian `A = df/dx` and input Jacobian `B = df/du` by central differences, provides longitudinal/lateral subsystem slices, and integrates perturbation responses with RK4.
- `src/aircraft6dof/modes.py` classifies longitudinal short-period and phugoid roots and lateral-directional Dutch-roll, roll-subsidence and spiral roots when eigenvalue patterns support those classifications.
- `src/aircraft6dof/equations.py` separates the rigid-body equations into `rigid_body_derivative()`. Aerodynamic/propulsive loads and environment effects remain assembled by `state_derivative()`.
- `src/aircraft6dof/aero.py` adds `CD_alpha`, `CY_p` and `CY_r`. Their defaults are zero so existing generic parameter sets do not silently acquire unsupported aircraft-specific coefficient values.
- `studies/adverse_yaw/modes.py` now calls the shared trim, linearization and mode APIs rather than carrying an independent lateral Jacobian.

## Representative mode results

For the generic aircraft at 55 m/s and 1,000 m in still air, after the current CG moment translation:

| Mode | Eigenvalue (1/s) | Additional metrics |
|---|---|---|
| Short period | -2.104 ± 2.865j | `wn=3.555 rad/s`, `zeta=0.592` |
| Phugoid | -0.01213 ± 0.20643j | `wn=0.2068 rad/s`, `zeta=0.0587`, period 30.44 s |
| Dutch roll | -0.89255 ± 3.63446j | `wn=3.7425 rad/s`, `zeta=0.2385`, period 1.729 s |
| Roll subsidence | -8.12764 | time constant 0.123 s |
| Spiral | +0.016848 | time to double about 41.1 s |

These are local mathematical results from the current generic coefficients. They are not evidence of aircraft-specific validation. The Euler-state formulation is singular near pitch ±90°.

## Verification coverage

The tests exercise Euler-rate conversion, trim equilibrium in the Euler form, Jacobian step-size sensitivity, subsystem slicing/coupling, physically plausible mode shapes, linear-versus-nonlinear small-input response, convergence as perturbation size shrinks, and the new derivative terms. A direct test checks that `rigid_body_derivative()` applies force, moment, gravity, inertia coupling and quaternion kinematics as expected.

The current branch also retains the CG reference-point-to-centre-of-gravity moment regression, trim-consistent main-run initialization, fixed-environment reporting checks, and Dryden turbulence regression coverage.

## Explicitly deferred

- No JSBSim dependency, model wrapper or external-comparison tests are added by this commit.
- No claim is made that the generic aircraft coefficients reproduce a specific aircraft.
- The adverse-yaw study's checked-in plots, animation and JSON summary predate the current CG moment translation; regenerate them before using them as evidence for the current model revision.
