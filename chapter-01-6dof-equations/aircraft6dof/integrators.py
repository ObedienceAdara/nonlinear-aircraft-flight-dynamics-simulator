"""Numerical integration utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rk4_step(
    state: NDArray[np.floating],
    dt: float,
    derivative,
) -> NDArray[np.float64]:
    """Advance a state by one fixed RK4 step.

    Parameters
    ----------
    state:
        Current state vector.
    dt:
        Positive integration time step in seconds.
    derivative:
        Callable with signature derivative(state) -> state derivative.

    Returns
    -------
    numpy.ndarray
        State after one RK4 step.
    """
    x = np.asarray(state, dtype=float)

    if x.ndim != 1:
        raise ValueError(f"state must be one-dimensional, got {x.ndim} dimensions.")
    if not np.isfinite(x).all():
        raise ValueError("state contains non-finite values.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a finite positive number.")

    k1 = np.asarray(derivative(x), dtype=float)
    k2 = np.asarray(derivative(x + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(derivative(x + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(derivative(x + dt * k3), dtype=float)

    if any(k.shape != x.shape for k in (k1, k2, k3, k4)):
        raise ValueError("derivative must return an array with the same shape as state.")
    if not all(np.isfinite(k).all() for k in (k1, k2, k3, k4)):
        raise ValueError("derivative returned non-finite values.")

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
