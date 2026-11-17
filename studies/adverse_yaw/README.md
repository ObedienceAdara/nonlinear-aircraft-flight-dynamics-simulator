# Adverse yaw study

Uses the repo's own `aircraft6dof` package (RK4 integration, quaternion
attitude, the repo's aero/actuator model) to reproduce and explain adverse
yaw: rolling right with ailerons alone first yaws the nose *left*, opposite
the turn, until sideslip builds up enough to bring the nose back around --
and how a rudder tied to the aileron input removes it.

## Run it

```bash
python studies/adverse_yaw/run_study.py   # from the repo root
```

Tests for the study (fast, no video) run with the rest of the suite:
`pytest tests/test_adverse_yaw_study.py`.

Takes ~2 minutes, almost all of it the animation render. Writes everything
to `outputs/`: seven PNGs, one MP4, and `summary.json` with every number used
in the plots.

## What's different from `main.py`'s default aircraft

One aero derivative: `Cn_da` (yaw due to aileron). The repo's default is
`+0.02`, which happens to be mildly **proverse** (yaw moment same sign as
the roll) -- there's no adverse yaw to see with that number. This study uses
`Cn_da = -0.03`, a plausible adverse value for a low-wing single-engine
type, purely so the effect exists to demonstrate. Nothing else about the
aircraft, environment, or integrator is changed.

The maneuver: trim for level flight at 55 m/s / 1000 m, then a clean 10°
right-aileron pulse from t=1.5s to t=3.0s, elevator and throttle held at
their trim values throughout, run for 14 s. Two cases, identical aileron
input in both:

- **Case A** -- aileron only, no rudder.
- **Case B** -- rudder commanded as `-k * aileron`, `k` chosen by the sweep
  below (the sign makes it "right aileron -> right/into-turn rudder").

The level-flight initialization now uses the shared `aircraft6dof.trim` solver rather than a study-specific Newton iteration, so the maneuver starts from the same trim machinery that the rest of the simulator can use.

## Where the yaw moment actually comes from

`Cn = Cn_da*da + Cn_p*p_hat + Cn_beta*beta + Cn_r*r_hat + Cn_dr*dr`

At the instant the aileron goes in, `beta = 0`, so the only two terms alive
are the aileron term (adverse, by construction here) and the roll-rate term
`Cn_p*p_hat` -- adverse-yaw yaw moment from a rolling wing is not just an
aileron drag effect, roll rate itself contributes through `Cn_p`. Both push
the nose the wrong way before sideslip has had any time to build up. As
sideslip grows, `Cn_beta*beta` grows with it and (being the opposite sign
here) drags the yaw rate back through zero and past it -- that crossing is the
little wiggle in case A's yaw-rate trace. Plot `03` shows all five terms
separately for both cases.

## Finding the rudder gain

`run_study.py` sweeps `k` from 0 to 1.5 and picks the value that minimizes
peak |sideslip| (plot `05`). Too little rudder: classic adverse yaw, nose
lags the turn. Too much: the nose now leads the turn the other way
(over-coordination) -- sideslip grows again, just with the opposite sign.
The minimum is a genuine minimum, not a monotonic "more rudder is better"
curve, which is worth seeing directly rather than taking on faith.

## A secondary effect worth noticing

Plot `02`'s third panel: sideslip also produces a rolling moment through
dihedral effect (`Cl_beta`), and in case A that moment opposes the
commanded roll for the first couple of seconds. Case A, despite having the
*same* aileron deflection as case B for the *same* duration, reaches a lower
maximum bank angle (38.65° versus 44.23° in the current stored run). Adverse
yaw doesn't just point the nose the wrong way -- the sideslip it creates eats
into the roll you asked for, too.


## Is the simulation consistent with its own linearization?

`modes.py` now delegates to the package's shared `trim()`, `linearize_trim()`
and `lateral_modes()` implementations. This removes the study's duplicate
finite-difference Jacobian and keeps the mode calculation on the same nonlinear
derivative used by the simulator.

For the study aircraft at 55 m/s and 1,000 m in still air, the current model gives:

| Mode | Root | Meaning |
|---|---|---|
| Dutch roll | -0.8926 +/- 3.6345j 1/s | damping ratio 0.2385, period 1.729 s |
| Roll subsidence | -8.1276 1/s | time constant 0.123 s |
| Spiral | +0.01685 1/s | weakly unstable, time to double about 41.1 s |

**Stored-output note:** the figures, animation, and `outputs/summary.json` were
refreshed in commit `8839080` after the CG-to-aerodynamic-reference moment
translation was already present in the model. They correspond to the study
configuration at that revision. Regenerate the outputs after changing the FDM,
trim solver, actuator model, or study setup before interpreting them as results
for a newer revision. The mode values above are calculated from the current
canonical implementation.

## How this differs from `main.py`'s environment

`main.py` flies through a steady 5 m/s wind and a 12 s gust, which would
muddy a clean roll-response comparison. This study uses still air, constant
density at 1000 m and gravity [0, 0, g] (down, NED). Gravity direction in
`main.py` was fixed upstream in PR #2; both now point down (the study uses 9.80665 m/s², `main.py` uses 9.806).

## Files

| File | What it does |
|---|---|
| `study_setup.py` | Aircraft/trim/maneuver setup, running a case, and the derived signals (alpha, beta, course, lateral g, per-term Cn/Cl breakdown) |
| `make_plots.py` | The seven static plots |
| `animate.py` | Side-by-side 3D comparison animation with a live yaw-rate/sideslip strip chart |
| `modes.py` | Thin wrapper over the canonical linearization and mode classifier |
| `run_study.py` | Runs everything and writes `outputs/` |

## Known limitation, stated plainly

`Cn_da = -0.03` is not identified from flight test or CFD for any specific
airframe -- it's a plausible adverse value chosen so the phenomenon exists
to study, exactly like the repo's own aero set is explicitly generic. Trust
the *mechanism and the relative comparison* (A vs. B, the gain sweep), not
the specific numeric value of any single-case sideslip angle as if it were
a real airplane's number.
