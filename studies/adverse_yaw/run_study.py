"""Entry point for the adverse yaw study.

    python3 run_study.py

Runs case A (aileron only) and case B (rudder-coordinated), sweeps the
rudder coordination gain to find the value that minimizes peak sideslip,
writes six engineering plots, a 3D comparison animation, and a JSON summary
of the numbers -- all under outputs/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import study_setup as S
import make_plots as P
import animate as V
import modes as M

OUT = Path(__file__).resolve().parent / "outputs"


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Running case A (aileron only)...")
    A = S.run_case("A: aileron only", 0.0)
    print("Running case B (rudder coordinated)...")
    B = S.run_case("B: rudder coordinated", S.BEST_K_DEFAULT)

    print("Sweeping rudder coordination gain...")
    ks = np.arange(0.0, 1.55, 0.05)
    rows = S.sweep_rudder_gain(ks)
    k_best = rows[int(np.argmin([r["peak_abs_beta"] for r in rows]))]["k"]
    print(f"  best k by minimum peak |sideslip|: {k_best:.2f}")

    print("Lateral-directional mode analysis...")
    modes = M.analyze()
    print(f"  Dutch roll: zeta = {modes['dutch_roll']['zeta']:.3f}, period = {modes['dutch_roll']['period_s']:.2f} s")

    print("Writing plots...")
    P.make_all(A, B, rows, k_best, OUT, modes)

    print("Rendering 3D comparison animation (this is the slow part)...")
    V.make_video(A, B, path=str(OUT / "adverse_yaw_3d.mp4"), fps=30)

    summary = {
        "case_A_aileron_only": S.metrics(A),
        "case_B_rudder_coordinated": S.metrics(B),
        "rudder_gain_sweep_best_k": k_best,
        "lateral_modes": {
            "dutch_roll": {k: (float(v) if not isinstance(v, complex) else [v.real, v.imag]) for k, v in modes["dutch_roll"].items()},
            "real_roots_1_per_s": modes["real_roots"],
        },
        "rudder_gain_sweep": rows,
        "maneuver": {
            "aileron_deg": S.AILERON_DEG,
            "roll_on_s": S.T_ROLL_ON_S,
            "roll_off_s": S.T_ROLL_OFF_S,
            "trim_airspeed_m_s": S.V_TRIM_M_S,
            "altitude_m": S.ALT_M,
            "Cn_da_used": S.CN_DA_ADVERSE,
            "Cn_da_repo_default": 0.02,
        },
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {time.time()-t0:.1f}s. Outputs in {OUT}")
    for p in sorted(OUT.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
