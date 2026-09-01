# Validation Plan

The canonical FDM should be validated in layers.

## Level 1 — mathematical invariants

- DCM orthogonality and determinant = 1.
- Quaternion norm remains 1 after propagation.
- Zero load/zero rate limiting cases.
- Rigid-body rotational coupling signs.
- Correct dimensions and finite values.

## Level 2 — environment

- Sea-level standard-atmosphere values.
- Atmospheric monotonicity with altitude.
- Gravity variation with latitude/altitude.
- Gust boundary and peak behavior.
- Reproducible stochastic turbulence with a fixed seed.

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

## Level 5 — independent comparison

Compare the simulator against:

- wind-tunnel or flight-test data;
- published aerodynamic databases;
- validated simulation software;
- independent analytical calculations.

A simulator that merely produces plausible-looking plots is not validated.

CI validation note: temporary pull-request validation branch.
