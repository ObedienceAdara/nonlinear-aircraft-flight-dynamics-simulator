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

## Current milestone

This document records the trim interface and numerical conventions for the
current repository milestone. Linearization and modal analysis will build on
the same trim result in the next milestone.
