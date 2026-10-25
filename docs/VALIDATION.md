# Validation Plan

The canonical FDM should be validated in layers.

## Level 1 — mathematical invariants

- DCM orthogonality and determinant = 1.
- Quaternion norm remains 1 after propagation.
- Zero load/zero rate limiting cases.
- Rigid-body rotational coupling signs.
- Aerodynamic reference-point to CG moment translation (r x F).
- Correct dimensions and finite values.

## Level 2 — environment

- Sea-level standard-atmosphere values.
- Atmospheric monotonicity with altitude.
- Gravity variation with latitude/altitude.
- Gust boundary and peak behavior.
- Reproducible stochastic turbulence with a fixed seed.
- Reported atmospheric properties match the environment used by the dynamics.
- Time-dependent deterministic controls/environment are evaluated at the RK4 stage times; a grid-aligned input step does not leak into the preceding interval.
- Timestep-refinement convergence for deterministic gust responses (for example compare dt and dt/2 using state and response metrics).
- Dryden output RMS and frequency-domain shapes match independent references within documented tolerances.

## Level 3 — aerodynamic primitives

- Force/moment scaling with dynamic pressure.
- Correct nondimensional rate scaling.
- Symmetry checks with zero sideslip/control input.
- Stability derivative sign checks.
- Wind/body transformation round trips.

## Level 4 — aircraft behavior

Once aircraft data are supplied:

- trim at specified airspeed/altitude;
- elevator trim and longitudinal static stability;
- lateral/directional static stability;
- short-period and phugoid response;
- Dutch-roll response;
- roll subsidence;
- spiral mode;
- turn performance and load factor;
- stall/post-stall behavior where modeled.

## Status

- **Trim:** implemented in `src/aircraft6dof/trim.py` and covered by dedicated tests for level flight, climb/descent, coordinated turns, sideslip and steady wind. The default `main.py` scenario starts from a trim and holds its baseline controls/actuator positions.
- **Time-dependent inputs:** deterministic callbacks are evaluated at RK4 stage times, with regression coverage for smooth forcing and grid-aligned command discontinuities. Timestep-convergence studies for the full gust-driven demo remain a separate validation task.
- **Dryden turbulence:** the discrete filter's spectral shapes and fixed-seed RMS behavior are tested, but `main.py` does not instantiate it. A future integration must update the stateful filter once per turbulence interval and hold/cache its sample across aircraft RK4 stages; full standard-severity parameterization and independent spectral comparison remain open validation tasks.
- **Linearization and modes:** central-difference A/B construction, Jacobian step-size sensitivity, trim equilibrium, subsystem structure, mode plausibility, and small-perturbation comparison are tested. These checks verify software behavior; they do not validate generic coefficients against an actual aircraft.
- **JSBSim code-to-code comparison:** implemented for rigid-body EOM, trim, local modes, and small-input time responses at one Cessna 172 operating point. Aerodynamic derivatives are locally fitted from JSBSim, so this is not independent aerodynamic validation. Wind-tunnel/flight-test comparison and multi-condition validation remain open.

## Level 5 — independent comparison

Compare the simulator against:

- wind-tunnel or flight-test data;
- published aerodynamic databases;
- validated simulation software;
- independent analytical calculations.

A simulator that merely produces plausible-looking plots is not validated.
