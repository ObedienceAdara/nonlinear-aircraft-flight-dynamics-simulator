# Comparison with JSBSim

This comparison exercises the simulator against JSBSim's Cessna 172 at one documented operating point. It separates three questions:

1. **Equations of motion:** given JSBSim's own applied loads, do the rigid-body accelerations, kinematics, and frames agree after accounting for the Earth-rotation terms omitted by this repository?
2. **Trim and local modes:** when the repository's compact aerodynamic model is fitted locally to JSBSim's coefficients, do trim and linearized mode estimates agree?
3. **Time response:** do both models respond similarly to the same small elevator, aileron, and rudder perturbations?

The generated report is `results/REPORT.md`; numeric data and the compared matrices are in `results/summary.json`. The report includes SI-valued maximum and RMS residuals as well as percentage errors normalized by the reference maneuver's peak change.

## Run

Install the optional dependency and run the suite:

```bash
python -m pip install -e ".[dev,compare]"
pytest
python comparison/jsbsim/run_comparison.py
```

The script typically takes tens of seconds. It writes four plots, the machine-readable summary, and the Markdown report to `comparison/jsbsim/results/`. The runtime-generated aircraft XML is written under `comparison/jsbsim/build/` and is ignored by Git.

## Fair-comparison design

The stock JSBSim `c172x` model includes a propulsion system, flight-control logic, autopilot/system files, and asymmetric payload details. The harness creates a controlled runtime copy with a symmetric payload, direct surface-angle inputs, and fixed thrust applied through the **payload-loaded** centre of gravity. The loaded CG is recomputed from the empty-aircraft weight/CG and the replacement 686 lb payload; the nominal source empty CG is not reused. It retains the stock aerodynamic database. The derived aircraft XML and aircraft data are not committed.

The repository's aerodynamic model has a smaller derivative set than JSBSim's tabular model. It estimates CLq and Cmq with alpha-dot as a diagnostic nuisance predictor, but only the rate derivative is adjusted in the deployed fit; the base static/control coefficients are retained from the supported-term fit to avoid moving trim as an artefact of local collinearity. The compact model still does not apply alpha-dot forces or moments, so the reported alpha-dot coefficients remain diagnostic-only. Drag fitting includes the fitted CD_alpha2 term and an explicit, default-zero CD_abs_beta term that the simulator now applies. The reported applied-model CD R² therefore includes the sideslip term; a separate ablation reports the score without that term. The alpha-dot on/off experiment keeps the repository coefficient set fixed across both JSBSim runs; an independent refit is stored only for diagnosis.

The reference condition is straight, level flight at **55 m/s and 1,000 m**, at the equator with heading north, in still air. This choice makes Coriolis acceleration zero in the straight reference case. The repository's active equations do not include full rotating-Earth terms, so Level 1 reports raw differences and differences after an analytical correction. Translational acceleration is corrected for rotating-Earth terms, and angular acceleration is transformed between the repository's inertial convention and JSBSim's ECEF-relative PQR convention. Levels 2 and 3 use the corresponding effective gravity.

## Outputs

| File | Purpose |
|---|---|
| `make_model.py` | Builds the controlled runtime JSBSim aircraft definition |
| `jsb.py` | Wraps JSBSim with SI inputs/outputs |
| `identify_aero.py` | Samples JSBSim and fits the repository's coefficient structure |
| `models.py` | Builds the repository model with matched mass properties and environment |
| `compare.py` | Runs EOM, trim/mode, and time-response comparisons |
| `plots.py`, `report.py` | Writes engineering plots and the report |
| `run_comparison.py` | Orchestrates the complete run |
| `results/summary.json` | Detailed numeric outputs including A/B matrices and error metrics |

## Interpretation and limits

This is a **code-to-code comparison**, not evidence that either model reproduces a real aircraft. The comparison uses JSBSim-derived coefficients and one local flight condition. The coefficient fit does not represent behavior across the flight envelope. The current model has no \(\dot{\alpha}\) terms, which is investigated with a diagnostic JSBSim variant; this is not a general correction.

No conclusion is made about stall, post-stall behavior, large control inputs, other aircraft, wind-tunnel accuracy, or flight-test validity. A credible validation campaign needs independent aircraft data, sensitivity studies, broader flight conditions, and documented acceptance criteria.

## Licensing

JSBSim's software and aircraft-data licensing is distinct from this repository's MIT license. The source C172 definition is loaded from the installed JSBSim package and transformed at runtime; the derived XML is not redistributed in this repository. The committed report and plots are experiment outputs, not copies of the source aircraft data.
