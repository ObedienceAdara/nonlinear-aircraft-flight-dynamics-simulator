"""Air-relative velocity and derived flight quantities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .frames import navigation_to_body_dcm
from .atmosphere import AtmosphereState


@dataclass(frozen=True)
class AirData:
    true_airspeed_m_s: float
    angle_of_attack_rad: float
    sideslip_rad: float
    dynamic_pressure_pa: float
    mach: float


def relative_velocity_body(
    velocity_body_m_s: np.ndarray,
    wind_ned_m_s: np.ndarray,
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """Return aircraft-relative air velocity resolved in body axes.

    Aircraft NED velocity is C_BN * velocity_body. Relative wind is aircraft
    velocity minus ambient wind, then transformed back to body axes.
    """
    vb = np.asarray(velocity_body_m_s, dtype=float)
    wn = np.asarray(wind_ned_m_s, dtype=float)

    if vb.shape != (3,) or wn.shape != (3,):
        raise ValueError("velocity_body_m_s and wind_ned_m_s must have shape (3,).")

    c_nb = navigation_to_body_dcm(phi, theta, psi)
    aircraft_ned = c_nb.T @ vb
    return c_nb @ (aircraft_ned - wn)


def compute_air_data(
    velocity_relative_body_m_s: np.ndarray,
    atmosphere: AtmosphereState,
) -> AirData:
    """Calculate V, alpha, beta, dynamic pressure and Mach."""
    vrel = np.asarray(velocity_relative_body_m_s, dtype=float)
    if vrel.shape != (3,):
        raise ValueError("velocity_relative_body_m_s must have shape (3,).")

    u_r, v_r, w_r = vrel
    V = float(np.linalg.norm(vrel))

    if V <= 1e-12:
        raise ValueError("Airspeed is too close to zero to define alpha/beta.")

    alpha = math.atan2(w_r, u_r)
    beta_argument = float(np.clip(v_r / V, -1.0, 1.0))
    beta = math.asin(beta_argument)
    q_dyn = 0.5 * atmosphere.density_kg_m3 * V**2
    mach = V / atmosphere.speed_of_sound_m_s

    return AirData(V, alpha, beta, q_dyn, mach)
