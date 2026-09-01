"""Aircraft Euler-angle kinematics for the 3-2-1 convention."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array3 = NDArray[np.float64]


def _finite_scalar(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def euler_angle_rates(
    phi: float,
    theta: float,
    p: float,
    q: float,
    r: float,
    *,
    singularity_tolerance: float = 1e-8,
) -> Array3:
    """Convert body angular rates [p, q, r] to 3-2-1 Euler-angle rates.

    Equations:
        phi_dot   = p + tan(theta) * (q*sin(phi) + r*cos(phi))
        theta_dot = q*cos(phi) - r*sin(phi)
        psi_dot   = (q*sin(phi) + r*cos(phi)) / cos(theta)

    Angles and angular rates are expressed in radians and rad/s.

    Raises
    ------
    ValueError
        If any argument is non-finite, tolerance is invalid, or theta is
        too close to the Euler singularity at +/- pi/2.
    """
    phi = _finite_scalar(phi, "phi")
    theta = _finite_scalar(theta, "theta")
    p = _finite_scalar(p, "p")
    q = _finite_scalar(q, "q")
    r = _finite_scalar(r, "r")

    if not np.isfinite(singularity_tolerance) or singularity_tolerance <= 0.0:
        raise ValueError("singularity_tolerance must be finite and positive.")

    cos_theta = np.cos(theta)
    if abs(cos_theta) < singularity_tolerance:
        raise ValueError(
            "Euler-angle singularity: cos(theta) is too close to zero."
        )

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)

    coupled_rate = q * sin_phi + r * cos_phi

    phi_dot = p + tan_theta * coupled_rate
    theta_dot = q * cos_phi - r * sin_phi
    psi_dot = coupled_rate / cos_theta

    return np.array([phi_dot, theta_dot, psi_dot], dtype=float)


def euler_rate_matrix(phi: float, theta: float, *, singularity_tolerance: float = 1e-8) -> NDArray[np.float64]:
    """Return T(phi, theta) such that eta_dot = T @ [p, q, r]."""
    phi = _finite_scalar(phi, "phi")
    theta = _finite_scalar(theta, "theta")

    cos_theta = np.cos(theta)
    if abs(cos_theta) < singularity_tolerance:
        raise ValueError(
            "Euler-angle singularity: cos(theta) is too close to zero."
        )

    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_theta = np.tan(theta)

    return np.array(
        [
            [1.0, sin_phi * tan_theta, cos_phi * tan_theta],
            [0.0, cos_phi, -sin_phi],
            [0.0, sin_phi / cos_theta, cos_phi / cos_theta],
        ],
        dtype=float,
    )
