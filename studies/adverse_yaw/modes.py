"""Lateral-directional modes at the study's trim point.

Uses the package's linearization (aircraft6dof.linearize) and mode classifier
(aircraft6dof.modes). The study previously carried its own 4-state Jacobian; the
numbers it reports are unchanged.
"""

from __future__ import annotations

import numpy as np

import study_setup as S
from aircraft6dof import TrimCondition, linearize_trim
from aircraft6dof.trim import trim
from aircraft6dof.modes import lateral_modes


def analyze(cn_da: float = S.CN_DA_ADVERSE) -> dict:
    aircraft = S.build_aircraft(cn_da)
    env = S.make_environment()
    tr = trim(aircraft, env, TrimCondition(airspeed_m_s=S.V_TRIM_M_S, altitude_m=S.ALT_M))
    lin = linearize_trim(aircraft, env, tr)
    lat = lin.lateral                                   # states v, p, r, phi
    found = lateral_modes(lin)

    dr = found["dutch_roll"]
    return {
        "jacobian": lat.A.tolist(),
        "trim_residual": tr.accel_residual.tolist(),
        "eigenvalues": [complex(e) for e in np.linalg.eigvals(lat.A)],
        "dutch_roll": {"lambda": dr.eigenvalue, "wn_rad_s": dr.wn_rad_s, "zeta": dr.zeta, "period_s": dr.period_s},
        "real_roots": sorted([found["roll_subsidence"].eigenvalue.real, found["spiral"].eigenvalue.real]),
    }


if __name__ == "__main__":
    m = analyze()
    print("eigenvalues:", np.round(m["eigenvalues"], 4))
    print("Dutch roll:", m["dutch_roll"])
    print("real roots:", m["real_roots"])