"""Shared setup for the adverse yaw study: aircraft, trim, the roll maneuver, derived signals.

Builds on the repo's own aircraft6dof package. The only aero number changed from
main.py is Cn_da (see README) -- everything else is the repo's aircraft as-is.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import main as repo_main  # the repo's own aircraft + actuator numbers
from aircraft6dof import AircraftModel, AircraftState, ControlInput, Environment, Simulator
from aircraft6dof.atmosphere import standard_atmosphere
from aircraft6dof.equations import AircraftParameters
from aircraft6dof.mathutils import dcm_body_to_ned_from_quat, quat_from_euler321

G = 9.80665
V_TRIM_M_S = 55.0
ALT_M = 1000.0

CN_DA_ADVERSE = -0.03   # repo default is +0.02, which is proverse (no adverse yaw at all)
AILERON_DEG = 10.0      # right roll
T_ROLL_ON_S = 1.5
T_ROLL_OFF_S = 3.0
BEST_K_DEFAULT = 0.55   # from the fine rudder-gain sweep; overwritten by run_study.py's own sweep
DURATION_S = 14.0
DT_S = 0.01


def build_aircraft(cn_da: float = CN_DA_ADVERSE) -> AircraftModel:
    base = repo_main.build_aircraft().parameters
    aero = replace(base.aero, Cn_da=cn_da)
    return AircraftModel(AircraftParameters(geometry=base.geometry, aero=aero, propulsion=base.propulsion))


def make_environment() -> Environment:
    # Still air, constant density, gravity down (0, 0, g). Unlike main.py's environment()
    # there is no wind or gust, so the only thing driving the response is the control input.
    atm = standard_atmosphere(ALT_M)
    return Environment(
        density_kg_m3=atm.density_kg_m3,
        speed_of_sound_m_s=atm.speed_of_sound_m_s,
        gravity_ned_m_s2=np.array([0.0, 0.0, G]),
    )


@dataclass(frozen=True)
class Trim:
    alpha_rad: float
    elevator_rad: float
    throttle: float


def trim_level_flight(aircraft: AircraftModel, env: Environment, V: float = V_TRIM_M_S) -> Trim:
    """Newton solve for steady level flight: udot = wdot = qdot = 0 with theta = alpha."""
    x = np.array([0.03, -0.01, 0.25])   # alpha, elevator, throttle

    def resid(x):
        a, de, th = x
        s = AircraftState(np.array([0.0, 0.0, -ALT_M]), np.array([V * np.cos(a), 0.0, V * np.sin(a)]),
                          np.zeros(3), quat_from_euler321(0.0, a, 0.0))
        d = aircraft.derivative(s, ControlInput(0.0, de, 0.0, th), env)
        return np.array([d.velocity_body_m_s[0], d.velocity_body_m_s[2], d.omega_body_rad_s[1]])

    for _ in range(40):
        r = resid(x)
        if np.linalg.norm(r) < 1e-10:
            return Trim(*map(float, x))
        J = np.empty((3, 3))
        for j in range(3):
            dx = np.zeros(3)
            dx[j] = 1e-6
            J[:, j] = (resid(x + dx) - r) / 1e-6
        x = x - np.linalg.solve(J, r)
    raise RuntimeError("trim did not converge")


def initial_state(trim: Trim, V: float = V_TRIM_M_S) -> AircraftState:
    a = trim.alpha_rad
    return AircraftState(
        position_ned_m=np.array([0.0, 0.0, -ALT_M]),
        velocity_body_m_s=np.array([V * np.cos(a), 0.0, V * np.sin(a)]),
        omega_body_rad_s=np.zeros(3),
        quaternion_bn=quat_from_euler321(0.0, a, 0.0),
    )


def make_actuators(trim: Trim):
    act = repo_main.build_actuators()
    act.elevator.position_rad = trim.elevator_rad   # start already at the trim deflection
    return act


def make_controls(trim: Trim, k_ari: float):
    """Same right-aileron pulse every time. Rudder is tied to it: rudder = -k * aileron
    (negative rudder = trailing edge right = 'right rudder' in this sign convention)."""
    da_cmd = np.deg2rad(AILERON_DEG)

    def controls(t: float) -> ControlInput:
        da = da_cmd if T_ROLL_ON_S <= t < T_ROLL_OFF_S else 0.0
        return ControlInput(aileron=da, elevator=trim.elevator_rad, rudder=-k_ari * da, throttle=trim.throttle)

    return controls


@dataclass
class CaseResult:
    name: str
    k_ari: float
    t: np.ndarray
    state: np.ndarray            # (n, 13)
    euler_deg: np.ndarray        # phi, theta, psi
    rates_deg: np.ndarray        # p, q, r
    alpha_deg: np.ndarray
    beta_deg: np.ndarray
    airspeed: np.ndarray
    course_deg: np.ndarray       # ground course chi
    cmd: np.ndarray              # aileron, elevator, rudder, throttle (commanded)
    act: np.ndarray              # aileron, elevator, rudder (actual, rad)
    n_y_g: np.ndarray            # lateral specific force / g  (what the slip ball feels)
    cn_terms: dict
    cl_terms: dict


def run_case(name: str, k_ari: float, cn_da: float = CN_DA_ADVERSE, duration_s: float = DURATION_S) -> CaseResult:
    aircraft = build_aircraft(cn_da)
    env = make_environment()
    trim = trim_level_flight(aircraft, env)
    hist = Simulator(aircraft).run(
        initial_state(trim), make_controls(trim, k_ari), lambda t: env,
        duration_s=duration_s, dt_s=DT_S, actuators=make_actuators(trim),
    )
    return _derive(name, k_ari, aircraft, env, hist)


def _derive(name, k_ari, aircraft, env, hist) -> CaseResult:
    X = hist.state
    t = hist.time_s
    vb, om = X[:, 3:6], X[:, 6:9]
    V = np.linalg.norm(vb, axis=1)
    alpha = np.arctan2(vb[:, 2], vb[:, 0])
    beta = np.arcsin(np.clip(vb[:, 1] / V, -1.0, 1.0))

    course = np.empty(len(t))
    n_y = np.empty(len(t))
    for i in range(len(t)):
        C = dcm_body_to_ned_from_quat(X[i, 9:13])
        v_ned = C @ vb[i]
        course[i] = np.degrees(np.arctan2(v_ned[1], v_ned[0]))
        s = AircraftState(X[i, :3], X[i, 3:6], X[i, 6:9], X[i, 9:13])
        u = ControlInput(*hist.actuator_deflection_rad[i], hist.control_command_rad[i, 3])
        d = aircraft.derivative(s, u, env)
        f_body = d.velocity_body_m_s + np.cross(om[i], vb[i]) - C.T @ env.gravity_ned_m_s2
        n_y[i] = f_body[1] / G

    aero = aircraft.parameters.aero
    b = aircraft.parameters.geometry.wing_span_m
    da, dr = hist.actuator_deflection_rad[:, 0], hist.actuator_deflection_rad[:, 2]
    p_hat = om[:, 0] * b / (2 * V)
    r_hat = om[:, 2] * b / (2 * V)
    cn_terms = {
        "aileron  Cn_da*da": aero.Cn_da * da,
        "roll rate  Cn_p*p^": aero.Cn_p * p_hat,
        "sideslip  Cn_beta*beta": aero.Cn_beta * beta,
        "yaw rate  Cn_r*r^": aero.Cn_r * r_hat,
        "rudder  Cn_dr*dr": aero.Cn_dr * dr,
    }
    cn_terms["total Cn"] = sum(cn_terms.values())
    cl_terms = {
        "aileron  Cl_da*da": aero.Cl_da * da,
        "dihedral  Cl_beta*beta": aero.Cl_beta * beta,
        "roll damping  Cl_p*p^": aero.Cl_p * p_hat,
    }

    return CaseResult(
        name=name, k_ari=k_ari, t=t, state=X,
        euler_deg=np.degrees(hist.euler_rad), rates_deg=np.degrees(om),
        alpha_deg=np.degrees(alpha), beta_deg=np.degrees(beta), airspeed=V, course_deg=course,
        cmd=hist.control_command_rad, act=hist.actuator_deflection_rad,
        n_y_g=n_y, cn_terms=cn_terms, cl_terms=cl_terms,
    )


def metrics(c: CaseResult) -> dict:
    r = c.rates_deg[:, 2]
    i_dip = int(np.argmin(r))
    i_pk = int(np.argmax(np.abs(c.beta_deg)))
    return {
        "name": c.name,
        "k_ari": float(c.k_ari),
        "adverse_yaw_rate_dip_deg_s": float(r[i_dip]),
        "adverse_yaw_dip_time_s": float(c.t[i_dip]),
        "peak_sideslip_deg": float(c.beta_deg[i_pk]),
        "peak_sideslip_time_s": float(c.t[i_pk]),
        "max_roll_rate_deg_s": float(c.rates_deg[:, 0].max()),
        "max_bank_deg": float(c.euler_deg[:, 0].max()),
        "peak_lateral_accel_g": float(c.n_y_g[np.argmax(np.abs(c.n_y_g))]),
        "altitude_change_m": float(-(c.state[-1, 2] - c.state[0, 2])),
        "final_airspeed_m_s": float(c.airspeed[-1]),
    }


def sweep_rudder_gain(ks: np.ndarray, duration_s: float = DURATION_S):
    rows = []
    for k in ks:
        c = run_case(f"k={k:.2f}", float(k), duration_s=duration_s)
        rows.append({
            "k": float(k),
            "peak_abs_beta": float(np.max(np.abs(c.beta_deg))),
            "max_beta": float(c.beta_deg.max()),
            "min_beta": float(c.beta_deg.min()),
            "yaw_dip": float(-min(0.0, c.rates_deg[:, 2].min())),
            "max_bank": float(c.euler_deg[:, 0].max()),
        })
    return rows
