from __future__ import annotations

import numpy as np

from .mathutils import normalize_quaternion
from .state import AircraftState


class RK4StateError(RuntimeError):
    """Raised when an RK4 step produces an invalid aircraft-state field."""


def _raise_if_nonfinite(state: AircraftState, time_s: float | None = None, step_index: int | None = None) -> None:
    fields = (
        ("position_ned_m", state.position_ned_m),
        ("velocity_body_m_s", state.velocity_body_m_s),
        ("omega_body_rad_s", state.omega_body_rad_s),
        ("quaternion_bn", state.quaternion_bn),
    )
    for name, value in fields:
        finite = np.isfinite(value)
        if not np.all(finite):
            bad_indices = np.flatnonzero(~finite).tolist()
            time_text = f"t={time_s:.6f} s" if time_s is not None else "time=unknown"
            step_text = f", step={step_index}" if step_index is not None else ""
            raise RK4StateError(
                f"RK4 produced a non-finite {name} field at {time_text}{step_text}; "
                f"invalid indices={bad_indices}, values={np.asarray(value).tolist()}"
            )


def add(x, dx, a):
    return AircraftState(
        x.position_ned_m + a * dx.position_ned_m,
        x.velocity_body_m_s + a * dx.velocity_body_m_s,
        x.omega_body_rad_s + a * dx.omega_body_rad_s,
        x.quaternion_bn + a * dx.quaternion_bn,
    )


def rk4_step(x, dt, f, time_s: float | None = None, step_index: int | None = None):
    k1 = f(x)
    k2 = f(add(x, k1, dt / 2))
    k3 = f(add(x, k2, dt / 2))
    k4 = f(add(x, k3, dt))

    q_candidate = x.quaternion_bn + dt * (
        k1.quaternion_bn + 2 * k2.quaternion_bn + 2 * k3.quaternion_bn + k4.quaternion_bn
    ) / 6
    candidate = AircraftState(
        x.position_ned_m + dt * (k1.position_ned_m + 2 * k2.position_ned_m + 2 * k3.position_ned_m + k4.position_ned_m) / 6,
        x.velocity_body_m_s + dt * (k1.velocity_body_m_s + 2 * k2.velocity_body_m_s + 2 * k3.velocity_body_m_s + k4.velocity_body_m_s) / 6,
        x.omega_body_rad_s + dt * (k1.omega_body_rad_s + 2 * k2.omega_body_rad_s + 2 * k3.omega_body_rad_s + k4.omega_body_rad_s) / 6,
        q_candidate,
    )
    _raise_if_nonfinite(candidate, time_s=time_s, step_index=step_index)

    try:
        q = normalize_quaternion(candidate.quaternion_bn)
    except ValueError as exc:
        time_text = f"t={time_s:.6f} s" if time_s is not None else "time=unknown"
        step_text = f", step={step_index}" if step_index is not None else ""
        raise RK4StateError(
            f"RK4 produced an invalid quaternion field at {time_text}{step_text}; "
            f"quaternion={candidate.quaternion_bn.tolist()}"
        ) from exc

    return AircraftState(
        candidate.position_ned_m,
        candidate.velocity_body_m_s,
        candidate.omega_body_rad_s,
        q,
    )
