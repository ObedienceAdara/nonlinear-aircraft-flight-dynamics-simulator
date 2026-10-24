# Repository vs JSBSim: Cessna 172 comparison

**Recorded result snapshot from the comparison harness.** CI reruns the experiment and publishes fresh outputs for its commit.

**Condition:** 55 m/s, 1,000 m, equator, heading north, still air.  
**Reference:** JSBSim C172 with symmetric payload, direct surface-angle inputs, and fixed external thrust.  
**Mass / wing:** 970.7 kg / 16.17 m²; span 10.97 m.

## What this comparison tests

Three layers are compared: rigid-body equations using identical JSBSim-applied loads, trim and local modes, and responses to small elevator, aileron, and rudder inputs.

For trim and time response, the repository's compact derivative model is fitted locally to JSBSim's C172 aerodynamic database. This helps isolate dynamics and implementation differences; **it does not independently validate the aerodynamic coefficients, and neither simulator is experimental ground truth.**

## Level 1 — equations of motion

At 40 randomized states, raw maximum translational-acceleration discrepancies were 0.0127, 0.0266, and 0.0407 m/s². After adding the analytical rotating-Earth acceleration terms omitted from the repository, they dropped to approximately 1.37e−13, 1.77e−13, and 1.12e−12 m/s².

Angular-acceleration discrepancies remained around 3.4e−05 to 5.0e−05 rad/s² and have not been independently isolated. Euler-rate and NED-velocity checks are included in summary.json.

## Level 2 — trim and modes

| Quantity | JSBSim | Repository | Difference |
|---|---:|---:|---:|
| Angle of attack | 0.386° | 0.379° | −0.006° |
| Elevator | 5.234° | 5.405° | +0.172° |
| Thrust | 1209.6 N | 1232.7 N | +23.1 N |

| Mode | JSBSim ωₙ | Repository ωₙ | JSBSim ζ | Repository ζ |
|---|---:|---:|---:|---:|
| Short period | 6.869 rad/s | 6.868 rad/s | 0.719 | 0.716 |
| Phugoid | 0.208 rad/s | 0.179 rad/s | 0.095 | 0.115 |
| Dutch roll | 2.817 rad/s | 2.818 rad/s | 0.209 | 0.210 |
| Roll subsidence | −10.521 s⁻¹ | −10.514 s⁻¹ | — | — |
| Spiral | −0.02142 s⁻¹ | −0.01957 s⁻¹ | — | — |

The phugoid is the main modal discrepancy. JSBSim includes lift and pitching-moment derivatives with respect to angle-of-attack rate, while the current repository model does not. With those JSBSim terms disabled diagnostically, the phugoid frequencies become 0.2083 and 0.2029 rad/s. The gap narrows, but this does not prove alpha-dot is the only source.

## Level 3 — time-response errors

Errors are relative to the peak change in the JSBSim reference signal. Each pair is worst pointwise error / RMS error.

| Input | Signal | Worst | RMS |
|---|---|---:|---:|
| Elevator doublet (±1°, 1 s each) | airspeed u | 43.1% | 20.3% |
| | pitch rate q | 13.9% | 0.9% |
| | pitch angle | 8.6% | 4.6% |
| | angle of attack | 7.8% | 0.9% |
| | altitude | 40.5% | 20.3% |
| Aileron pulse (+1°, 1.5 s) | lateral velocity v | 2.4% | 0.8% |
| | roll rate p | 10.4% | 0.6% |
| | yaw rate r | 1.7% | 0.9% |
| | bank angle | 2.3% | 1.3% |
| | sideslip | 2.4% | 0.8% |
| Rudder pulse (+1°, 1 s) | lateral velocity v | 3.5% | 1.1% |
| | roll rate p | 4.0% | 1.2% |
| | yaw rate r | 5.2% | 1.5% |
| | bank angle | 2.5% | 1.4% |
| | sideslip | 3.6% | 1.1% |

The speed and altitude response to the elevator doublet show the largest RMS differences, consistent with the phugoid mismatch being important. The full harness computes and records these metrics automatically.

## Figures

- modes_s_plane.png — eigenvalues of both models and the alpha-dot diagnostic.
- time_elevator_doublet.png — elevator doublet and residual histories.
- time_aileron_pulse.png — aileron pulse and residual histories.
- time_rudder_pulse.png — rudder pulse and residual histories.

## Limits

- One operating point and small control inputs; no stall, post-stall, high-angle, turning, or envelope-wide tests.
- Repository coefficients are fitted from JSBSim, not wind-tunnel or flight-test data.
- Rotating-Earth effects are corrected analytically only in Level 1; active repository EOM remain flat-Earth / non-rotating.
- The source C172 data retain their upstream license; derived XML is generated locally and not committed.

## Reproduce

    python -m pip install -e ".[dev,compare]"
    pytest
    python comparison/jsbsim/run_comparison.py
