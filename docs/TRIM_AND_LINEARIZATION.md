# Aircraft trim

## Trim

A trim is a state and control setting where the body-axis accelerations
`udot, vdot, wdot, pdot, qdot, rdot` are all zero while the flight condition is held.

```python
from aircraft6dof import TrimCondition
from aircraft6dof.trim import trim

r = trim(aircraft, env, TrimCondition(
    airspeed_m_s=55.0,
    altitude_m=1000.0,
    flight_path_rad=0.0,       # relative to the air mass; + is a climb
    turn_rate_rad_s=0.0,       # heading rate; + is a right turn
    sideslip_rad=0.0,
    heading_rad=0.0,
))
r.alpha_rad, r.controls, r.state, r.euler_rad, r.accel_residual
```

**Unknowns.** Straight symmetric flight solves for angle of attack, elevator and throttle (3 unknowns, using
`udot, wdot, qdot`). A turn or sideslip also solves for bank angle, aileron and rudder (6 unknowns, all six
accelerations). Pass `include_lateral=True` to force the 6-unknown form, for example for a model with an engine-out asymmetry.

**Pitch from flight-path angle.** With air-relative velocity components `u, v, w` the climb rate is
`V sin(gamma) = u sin(theta) - (v sin(phi) + w cos(phi)) cos(theta)`. That is `A sin(theta) - B cos(theta) = sin(gamma)`,
which has the closed-form solution `theta = atan2(B, A) + asin(sin(gamma) / sqrt(A^2 + B^2))`
(`pitch_from_flight_path`).

**Turn rates.** For heading rate `psi_dot`, the body rates are
`p = -psi_dot sin(theta)`, `q = psi_dot sin(phi) cos(theta)`, `r = psi_dot cos(phi) cos(theta)`.

**Solver.** Damped Gauss-Newton on the residual vector. The Jacobian is a central finite difference
(step 1e-6). Each step is `lstsq(J, -r)` followed by a backtracking line search on `||r||`. Nothing outside NumPy is used.
Typical convergence for this repo's aircraft is 2 to 3 iterations to a residual near 1e-15.

**Wind.** The state's body velocity is ground-relative, so a steady wind is added to the air-relative
velocity: `v_ground_body = v_air_body + C_bn^T wind`. This is only allowed for straight trims. In a turn the
body-frame ground velocity rotates, so the body accelerations cannot all be zero, and `trim()` raises `ValueError`.

**Failure.** `TrimError` is raised if the residual does not reach `tol` (default 1e-9). The exception carries the last
iterate in `.result`. A converged trim with throttle outside [0, 1] is returned with a warning in `result.warnings`,
because the propulsion model clips throttle and that trim could not be flown.

## Linearization

The nonlinear simulator represents attitude with a quaternion. For local analysis,
`linearize.py` uses the Euler state

```
[N, E, D, u, v, w, p, q, r, phi, theta, psi]
```

and the control vector `[aileron, elevator, rudder, throttle]`. It evaluates the
canonical `aircraft.derivative()` through an Euler-rate wrapper, then computes
`A = df/dx` and `B = df/du` using central finite differences. The default step
is `1e-6`, scaled by the magnitude of the component being perturbed. The resulting
`LinearModel` stores the operating point and nonlinear `xdot0`, exposes the
longitudinal and lateral subsystem matrices, and integrates perturbation responses
using RK4 with a zero-order-held input.

```python
from aircraft6dof import TrimCondition, linearize_trim
from aircraft6dof.trim import trim

tr = trim(aircraft, env, TrimCondition(55.0, altitude_m=1000.0))
model = linearize_trim(aircraft, env, tr)
print(model.A.shape, model.B.shape)  # (12, 12), (12, 4)
```

Linearization uses Euler angles, so it is a local model and is singular near
pitch \(\pm90^\circ\). It does not replace the quaternion nonlinear model.

## Flight-dynamics modes

`modes.py` classifies the eigenvalues of the longitudinal
\((u,w,q,\theta)\) and lateral-directional \((v,p,r,\phi)\) submodels.

| Mode | Representative root at 55 m/s, 1,000 m, still air | Interpretation |
|---|---|---|
| Short period | \(-2.104\pm2.865j\;\mathrm{s}^{-1}\) | Fast longitudinal oscillation; \(\zeta\approx0.592\) |
| Phugoid | \(-0.01213\pm0.20643j\;\mathrm{s}^{-1}\) | Slow speed/flight-path exchange; \(\zeta\approx0.0587\) |
| Dutch roll | \(-0.8926\pm3.6345j\;\mathrm{s}^{-1}\) | Lateral-directional oscillation; \(\zeta\approx0.2385\), period 1.729 s |
| Roll subsidence | \(-8.128\;\mathrm{s}^{-1}\) | Stable real mode; time constant 0.123 s |
| Spiral | \(+0.01685\;\mathrm{s}^{-1}\) | Weakly unstable real mode; time to double about 41.1 s |

These roots are numerical diagnostics for the current generic coefficients and operating
point. They are not validation against a real aircraft.

## Current status

The current repository supports nonlinear 6-DOF simulation, trim, local A/B
linearization, flight-dynamics mode classification, small-response comparisons,
and an optional JSBSim code-to-code comparison under `comparison/jsbsim/`.
The JSBSim harness compares equations of motion, trim/modes, and small-input
responses at one operating point. Its aerodynamic coefficients are fitted from
JSBSim, so it is not independent aerodynamic or real-aircraft validation.

See [the comparison guide](../comparison/jsbsim/README.md) and
[the validation plan](VALIDATION.md) for assumptions, reproduced results, and
remaining validation gaps.
