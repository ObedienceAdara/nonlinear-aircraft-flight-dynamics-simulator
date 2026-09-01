"""Local numerical integration utility for Chapter 02."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rk4_step(state: NDArray[np.floating], dt: float, derivative) -> NDArray[np.float64]:
    """Advance an arbitrary state with one classic fourth-order RK step."""
    x = np.asarray(state, dtype=float)
    if x.ndim != 1:
        raise ValueError("state must be one-dimensional.")
    if not np.isfinite(x).all():
        raise ValueError("state contains non-finite values.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive.")

    k1 = np.asarray(derivative(x), dtype=float)
    k2 = np.asarray(derivative(x + 0.5 * dt * k1), dtype=float)
    k3 = np.asarray(derivative(x + 0.5 * dt * k2), dtype=float)
    k4 = np.asarray(derivative(x + dt * k3), dtype=float)

    if not all(k.shape == x.shape for k in (k1, k2, k3, k4)):
        raise ValueError("derivative output must match state shape.")
    if not all(np.isfinite(k).all() for k in (k1, k2, k3, k4)):
        raise ValueError("derivative returned non-finite values.")

    return x + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
