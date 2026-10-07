"""Steady-state trim for the 6-DOF model.

A trim is a state and control setting for which the body-axis accelerations
(udot, vdot, wdot, pdot, qdot, rdot) are all zero while the flight condition
(airspeed, flight-path angle, turn rate, sideslip) is held at the requested
value. Straight level, climbing/descending, coordinated-turn and steady
sideslip trims are supported.

The solver is a damped Gauss-Newton iteration with a finite-difference
Jacobian, so it needs nothing beyond NumPy and works with any model that
exposes ``aircraft.derivative(state, controls, environment)``.

Conventions
-----------
* Euler angles are 3-2-1 (yaw, pitch, roll), body to NED.
* ``flight_path_rad`` is measured relative to the air mass.
* ``turn_rate_rad_s`` is heading rate, positive to the right.
* Steady wind is allowed for straight trims only. In a turn the body-frame
  *ground* velocity rotates, so the body-axis accelerations cannot all be zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mathutils import dcm_body_to_ned_from_quat, quat_from_euler321
from .state import AircraftState, ControlInput, Environment

G_REF = 9.80665

BASE_UNKNOWNS = ("alpha", "throttle", "elevator")
LATERAL_UNKNOWNS = ("phi", "aileron", "rudder")


class TrimError(RuntimeError):
    """Raised when the trim iteration fails; ``result`` holds the last iterate."""

    def __init__(self, message: str, result: "TrimResult | None" = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class TrimCondition:
    """Flight condition to trim to."""

    airspeed_m_s: float
    altitude_m: float = 0.0
    flight_path_rad: float = 0.0
    turn_rate_rad_s: float = 0.0
    sideslip_rad: float = 0.0
    heading_rad: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.airspeed_m_s) or self.airspeed_m_s <= 0.0:
            raise ValueError("airspeed_m_s must be positive and finite")

    @property
    def needs_lateral(self) -> bool:
        return self.turn_rate_rad_s != 0.0 or self.sideslip_rad != 0.0


@dataclass(frozen=True)
class TrimResult:
    condition: TrimCondition
    state: AircraftState
    controls: ControlInput
    alpha_rad: float
    beta_rad: float
    euler_rad: np.ndarray            # phi, theta, psi
    unknowns: tuple[str, ...]
    converged: bool
    iterations: int
    accel_residual: np.ndarray       # udot, vdot, wdot, pdot, qdot, rdot at the solution
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def residual_norm(self) -> float:
        return float(np.max(np.abs(self.accel_residual)))

    def summary(self) -> str:
        phi, theta, psi = np.rad2deg(self.euler_rad)
        c = self.controls
        lines = [
            f"Trim at V={self.condition.airspeed_m_s:.2f} m/s, h={self.condition.altitude_m:.0f} m "
            f"(gamma={np.rad2deg(self.condition.flight_path_rad):.2f} deg, "
            f"turn rate={np.rad2deg(self.condition.turn_rate_rad_s):.2f} deg/s)",
            f"  alpha={np.rad2deg(self.alpha_rad):.4f} deg  beta={np.rad2deg(self.beta_rad):.4f} deg",
            f"  phi={phi:.4f} deg  theta={theta:.4f} deg  psi={psi:.2f} deg",
            f"  aileron={np.rad2deg(c.aileron):.4f} deg  elevator={np.rad2deg(c.elevator):.4f} deg  "
            f"rudder={np.rad2deg(c.rudder):.4f} deg  throttle={c.throttle:.5f}",
            f"  converged={self.converged} in {self.iterations} iterations, "
            f"max |accel residual|={self.residual_norm:.2e}",
        ]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def pitch_from_flight_path(alpha: float, beta: float, phi: float, gamma: float) -> float:
    """Pitch angle theta that gives flight-path angle gamma (air-relative).

    With air-relative unit velocity (a, y, z) = (cos a cos b, sin b, sin a cos b),
    the climb rate is  V sin(gamma) = u sin(theta) - (v sin(phi) + w cos(phi)) cos(theta).
    That is  A sin(theta) - B cos(theta) = sin(gamma),  solved in closed form.
    """
    a = np.cos(alpha) * np.cos(beta)
    b = (np.sin(beta) * np.sin(phi) + np.sin(alpha) * np.cos(beta) * np.cos(phi))
    r = np.hypot(a, b)
    s = np.sin(gamma)
    if abs(s) > r:
        raise ValueError("flight-path angle not reachable at this alpha/beta/phi")
    return float(np.arctan2(b, a) + np.arcsin(s / r))


def _build(condition: TrimCondition, env: Environment, x: dict) -> tuple[AircraftState, ControlInput, float, float]:
    """Assemble the aircraft state and controls from the unknowns ``x``."""
    alpha = x["alpha"]
    beta = condition.sideslip_rad
    phi = x.get("phi", 0.0)
    theta = pitch_from_flight_path(alpha, beta, phi, condition.flight_path_rad)
    psi = condition.heading_rad
    V = condition.airspeed_m_s

    q_bn = quat_from_euler321(phi, theta, psi)
    v_air_b = V * np.array([np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)])
    wind = np.asarray(env.wind_ned_m_s, float)
    v_ground_b = v_air_b + dcm_body_to_ned_from_quat(q_bn).T @ wind

    rate = condition.turn_rate_rad_s
    omega = np.array([-rate * np.sin(theta),
                      rate * np.sin(phi) * np.cos(theta),
                      rate * np.cos(phi) * np.cos(theta)])

    state = AircraftState(
        position_ned_m=np.array([0.0, 0.0, -condition.altitude_m]),
        velocity_body_m_s=v_ground_b,
        omega_body_rad_s=omega,
        quaternion_bn=q_bn,
    )
    controls = ControlInput(
        aileron=x.get("aileron", 0.0),
        elevator=x["elevator"],
        rudder=x.get("rudder", 0.0),
        throttle=x["throttle"],
    )
    return state, controls, theta, psi


def _accelerations(aircraft, env: Environment, state: AircraftState, controls: ControlInput) -> np.ndarray:
    d = aircraft.derivative(state, controls, env)
    return np.concatenate([d.velocity_body_m_s, d.omega_body_rad_s])


def trim(
    aircraft,
    environment: Environment,
    condition: TrimCondition,
    initial_guess: dict | None = None,
    *,
    include_lateral: bool | None = None,
    tol: float = 1e-9,
    max_iter: int = 80,
) -> TrimResult:
    """Find a steady trim for ``condition``.

    ``include_lateral`` adds roll angle, aileron and rudder to the unknowns and
    uses all six acceleration equations. By default it is switched on only when
    the condition has a turn or sideslip. Use ``True`` for a model with lateral
    asymmetry (for example an engine-out case), where level flight needs
    aileron and rudder too.

    Raises ``TrimError`` if the iteration does not reach ``tol``.
    """
    if condition.needs_lateral and np.any(np.asarray(environment.wind_ned_m_s) != 0.0) and condition.turn_rate_rad_s != 0.0:
        raise ValueError("steady wind is only supported for straight (non-turning) trims")
    lateral = condition.needs_lateral if include_lateral is None else bool(include_lateral)
    names = BASE_UNKNOWNS + (LATERAL_UNKNOWNS if lateral else ())
    eq_idx = np.arange(6) if lateral else np.array([0, 2, 4])   # udot, wdot, qdot for the symmetric case

    guess = {"alpha": 0.05, "throttle": 0.5, "elevator": 0.0, "phi": 0.0, "aileron": 0.0, "rudder": 0.0}
    if condition.turn_rate_rad_s != 0.0:
        guess["phi"] = float(np.arctan(condition.turn_rate_rad_s * condition.airspeed_m_s / G_REF))
    if initial_guess:
        guess.update(initial_guess)
    x = np.array([guess[n] for n in names], float)

    def residual(vec: np.ndarray) -> np.ndarray:
        state, controls, _, _ = _build(condition, environment, dict(zip(names, vec)))
        return _accelerations(aircraft, environment, state, controls)[eq_idx]

    r = residual(x)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        if np.max(np.abs(r)) < tol:
            iterations -= 1
            break
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = 1e-6
            dx = np.zeros_like(x)
            dx[j] = h
            J[:, j] = (residual(x + dx) - residual(x - dx)) / (2 * h)
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        lam, r_new, x_new = 1.0, r, x
        for _ in range(20):                       # backtracking line search on ||r||
            x_try = x + lam * step
            try:
                r_try = residual(x_try)
            except ValueError:
                r_try = None
            if r_try is not None and np.linalg.norm(r_try) < np.linalg.norm(r):
                x_new, r_new = x_try, r_try
                break
            lam *= 0.5
        else:
            break                                 # no improving step
        x, r = x_new, r_new

    state, controls, theta, psi = _build(condition, environment, dict(zip(names, x)))
    full = _accelerations(aircraft, environment, state, controls)
    converged = bool(np.max(np.abs(r)) < tol)
    phi = dict(zip(names, x)).get("phi", 0.0)

    warnings = []
    if not 0.0 <= controls.throttle <= 1.0:
        warnings.append(f"throttle {controls.throttle:.3f} is outside [0, 1]; trim is not flyable")
    if not lateral and np.max(np.abs(full[[1, 3, 5]])) > 1e-6:
        warnings.append("lateral accelerations are non-zero; the model is asymmetric, use include_lateral=True")

    result = TrimResult(
        condition=condition, state=state, controls=controls,
        alpha_rad=float(dict(zip(names, x))["alpha"]), beta_rad=condition.sideslip_rad,
        euler_rad=np.array([phi, theta, psi]), unknowns=names, converged=converged,
        iterations=iterations, accel_residual=full, warnings=tuple(warnings),
    )
    if not converged:
        raise TrimError(
            f"trim did not converge: max |residual|={np.max(np.abs(r)):.3e} after {iterations} iterations "
            f"(unknowns: {', '.join(names)})", result)
    return result