"""Air-relative kinematics shared by the dynamics, the guard and the reporting.

Relative wind, airspeed, angle of attack and sideslip are computed here and
nowhere else, so the equations of motion, the simulation health checks and the
exported diagnostics cannot drift apart.

Two airspeed thresholds are defined, and they intentionally differ because they
protect against different problems:

``MIN_AIRSPEED_M_S``
    Divide-by-zero guard. At or below this airspeed the flow direction is
    undefined, so alpha and beta are reported as zero and the aerodynamic loads
    are zero.

``MIN_RATE_NORMALIZATION_AIRSPEED_M_S``
    Regularization of the nondimensional body rates (``p b / 2V``, ``q c / 2V``,
    ``r b / 2V``) in the coefficient buildup. Without it the rate terms grow
    without bound as V -> 0 and the damping derivatives dominate at speeds far
    below any regime the coefficients describe. It has no effect above this
    airspeed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mathutils import dcm_body_to_ned_from_quat

MIN_AIRSPEED_M_S = 1e-8
MIN_RATE_NORMALIZATION_AIRSPEED_M_S = 1.0


@dataclass(frozen=True)
class AirData:
    """Air-relative quantities for one state and environment.

    ``velocity_body_m_s`` is the velocity of the air relative to the aircraft's
    motion expressed in body axes, i.e. aircraft velocity minus the wind and gust.
    """

    velocity_body_m_s: np.ndarray
    airspeed_m_s: float
    alpha_rad: float
    beta_rad: float


def air_data(velocity_body_m_s, quaternion_bn, environment) -> AirData:
    """Relative wind, airspeed, alpha and beta.

    ``velocity_body_m_s`` is the aircraft's inertial velocity in body axes and
    ``quaternion_bn`` the body-to-NED attitude. The quaternion is normalized
    internally. Below ``MIN_AIRSPEED_M_S`` alpha and beta are zero.
    """
    C = dcm_body_to_ned_from_quat(quaternion_bn)
    wind_ned = np.asarray(environment.wind_ned_m_s, dtype=float) + np.asarray(environment.gust_ned_m_s, dtype=float)
    vrel_n = C @ np.asarray(velocity_body_m_s, dtype=float) - wind_ned
    vrel_b = C.T @ vrel_n
    V = float(np.linalg.norm(vrel_b))
    if V > MIN_AIRSPEED_M_S:
        alpha = float(np.arctan2(vrel_b[2], vrel_b[0]))
        beta = float(np.arcsin(np.clip(vrel_b[1] / V, -1.0, 1.0)))
    else:
        alpha = 0.0
        beta = 0.0
    return AirData(vrel_b, V, alpha, beta)
