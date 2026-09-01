"""3-2-1 Euler direction-cosine matrices for NED navigation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def body_to_navigation_dcm(phi: float, theta: float, psi: float) -> NDArray[np.float64]:
    """Map a vector from aircraft body axes to NED navigation axes."""
    sphi, cphi = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)
    spsi, cpsi = np.sin(psi), np.cos(psi)

    return np.array([
        [cth*cpsi, sphi*sth*cpsi-cphi*spsi, cphi*sth*cpsi+sphi*spsi],
        [cth*spsi, sphi*sth*spsi+cphi*cpsi, cphi*sth*spsi-sphi*cpsi],
        [-sth,     sphi*cth,                 cphi*cth],
    ], dtype=float)


def navigation_to_body_dcm(phi: float, theta: float, psi: float) -> NDArray[np.float64]:
    """Map a vector from NED navigation axes to aircraft body axes."""
    return body_to_navigation_dcm(phi, theta, psi).T
