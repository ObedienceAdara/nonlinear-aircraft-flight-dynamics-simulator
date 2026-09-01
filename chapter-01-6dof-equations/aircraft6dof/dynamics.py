"""Rigid-body translational and rotational 6-DOF dynamics.

State ordering:
    [u, v, w, p, q, r]

All quantities use SI units.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Vector3 = NDArray[np.float64]
State6 = NDArray[np.float64]


def _as_vector3(value: NDArray[np.floating] | list[float] | tuple[float, ...], name: str) -> Vector3:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}.")
    return array


def _as_inertia(value: NDArray[np.floating]) -> NDArray[np.float64]:
    inertia = np.asarray(value, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError(f"inertia must have shape (3, 3), got {inertia.shape}.")
    if not np.allclose(inertia, inertia.T, atol=1e-12):
        raise ValueError("inertia tensor must be symmetric.")
    if not np.all(np.linalg.eigvalsh(inertia) > 0.0):
        raise ValueError("inertia tensor must be positive definite.")
    return inertia


def equations_of_motion(
    state: NDArray[np.floating] | list[float] | tuple[float, ...],
    mass: float,
    inertia: NDArray[np.floating],
    force_body: NDArray[np.floating] | list[float] | tuple[float, ...],
    moment_body: NDArray[np.floating] | list[float] | tuple[float, ...],
) -> State6:
    """Compute translational and rotational body-frame state derivatives.

    The implemented equations are:

        m (v_dot + omega x v) = F
        I omega_dot + omega x (I omega) = M

    so:

        v_dot = F / m - omega x v
        omega_dot = solve(I, M - omega x (I omega))

    Parameters
    ----------
    state:
        [u, v, w, p, q, r].
    mass:
        Vehicle mass in kg.
    inertia:
        3x3 body-frame inertia tensor in kg m^2.
    force_body:
        Applied force [X, Y, Z] in N.
    moment_body:
        Applied moment [L, M, N] in N m.

    Returns
    -------
    numpy.ndarray
        [u_dot, v_dot, w_dot, p_dot, q_dot, r_dot].
    """
    x = np.asarray(state, dtype=float)
    if x.shape != (6,):
        raise ValueError(f"state must have shape (6,), got {x.shape}.")
    if not np.isfinite(x).all():
        raise ValueError("state contains non-finite values.")
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("mass must be a finite positive number.")

    inertia_matrix = _as_inertia(inertia)
    force = _as_vector3(force_body, "force_body")
    moment = _as_vector3(moment_body, "moment_body")

    velocity = x[0:3]
    angular_rate = x[3:6]

    velocity_dot = force / mass - np.cross(angular_rate, velocity)

    angular_momentum = inertia_matrix @ angular_rate
    angular_rate_dot = np.linalg.solve(
        inertia_matrix,
        moment - np.cross(angular_rate, angular_momentum),
    )

    return np.concatenate((velocity_dot, angular_rate_dot))
