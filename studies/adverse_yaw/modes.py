"""Lateral-directional modes at the trim point, from a central-difference Jacobian
of the repo's own aircraft.derivative(). Standard decoupled 4-state model:
(v, p, r, phi), with u, w, q and the trim attitude held fixed."""

from __future__ import annotations

import numpy as np

import study_setup as S
from aircraft6dof import AircraftState, ControlInput
from aircraft6dof.mathutils import quat_from_euler321


def lateral_jacobian(aircraft, env, trim):
    s0 = S.initial_state(trim)
    u0, w0 = s0.velocity_body_m_s[0], s0.velocity_body_m_s[2]
    theta0 = trim.alpha_rad
    ctrl = ControlInput(0.0, trim.elevator_rad, 0.0, trim.throttle)

    def f(x):
        v, p, r, phi = x
        st = AircraftState(s0.position_ned_m, np.array([u0, v, w0]), np.array([p, 0.0, r]),
                           quat_from_euler321(phi, theta0, 0.0))
        d = aircraft.derivative(st, ctrl, env)
        phidot = p + r * np.cos(phi) * np.tan(theta0)       # kinematics with q = 0
        return np.array([d.velocity_body_m_s[1], d.omega_body_rad_s[0], d.omega_body_rad_s[2], phidot])

    x0 = np.zeros(4)
    J = np.zeros((4, 4))
    steps = np.array([1e-3, 1e-6, 1e-6, 1e-6])
    for j in range(4):
        dx = np.zeros(4)
        dx[j] = steps[j]
        J[:, j] = (f(x0 + dx) - f(x0 - dx)) / (2 * steps[j])
    return J, f(x0)


def analyze(cn_da: float = S.CN_DA_ADVERSE) -> dict:
    aircraft = S.build_aircraft(cn_da)
    env = S.make_environment()
    trim = S.trim_level_flight(aircraft, env)
    J, resid = lateral_jacobian(aircraft, env, trim)
    ev = np.linalg.eigvals(J)
    out = {"jacobian": J.tolist(), "trim_residual": resid.tolist(), "eigenvalues": [complex(e) for e in ev]}
    osc = [e for e in ev if abs(e.imag) > 1e-6 and e.imag > 0]
    if osc:
        lam = osc[0]
        wn = abs(lam)
        out["dutch_roll"] = {"lambda": lam, "wn_rad_s": float(wn), "zeta": float(-lam.real / wn),
                             "period_s": float(2 * np.pi / lam.imag)}
    real = sorted([float(e.real) for e in ev if abs(e.imag) <= 1e-6])
    out["real_roots"] = real
    return out


if __name__ == "__main__":
    m = analyze()
    print("eigenvalues:", np.round(m["eigenvalues"], 4))
    print("Dutch roll:", m.get("dutch_roll"))
    print("real roots:", m["real_roots"])
