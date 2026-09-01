"""Frame transformations for a 3-2-1 (yaw-pitch-roll) Euler convention."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def body_to_navigation_dcm(phi: float, theta: float, psi: float) -> NDArray[np.float64]:
    """Return the NED navigation-to-body transpose convention used here.

    The returned matrix maps a vector expressed in body coordinates into NED
    navigation coordinates:

        v_n = C_bn @ v_b

    where the columns are the body basis vectors represented in NED.

    The sequence is yaw (psi), pitch (theta), roll (phi).
    """
    sphi, cphi = np.sin(phi), np.cos(phi)
    sth, cth = np.sin(theta), np.cos(theta)
    spsi, cpsi = np.sin(psi), np.cos(psi)

    return np.array(
        [
            [
                cth * cpsi,
                sphi * sth * cpsi - cphi * spsi,
                cphi * sth * cpsi + sphi * spsi,
            ],
            [
                cth * spsi,
                sphi * sth * spsi + cphi * cpsi,
                cphi * sth * spsi - sphi * cpsi,
            ],
            [
                -sth,
                sphi * cth,
                cphi * cth,
            ],
        ],
        dtype=float,
    )


def navigation_to_body_dcm(phi: float, theta: float, psi: float) -> NDArray[np.float64]:
    """Return the transpose rotation mapping NED vectors into body axes."""
    return body_to_navigation_dcm(phi, theta, psi).T
