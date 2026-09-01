"""Flat-Earth North-East-Down navigation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class FlatEarthNED:
    """Minimal flat-Earth NED position/velocity state."""

    north_m: float = 0.0
    east_m: float = 0.0
    down_m: float = 0.0

    def position(self) -> np.ndarray:
        return np.array([self.north_m, self.east_m, self.down_m], dtype=float)

    @property
    def altitude_m(self) -> float:
        return -self.down_m

    def integrate(self, velocity_ned: np.ndarray, dt: float) -> None:
        """Integrate NED position using current NED velocity."""
        velocity = np.asarray(velocity_ned, dtype=float)
        if velocity.shape != (3,):
            raise ValueError("velocity_ned must have shape (3,).")
        if not np.isfinite(velocity).all():
            raise ValueError("velocity_ned contains non-finite values.")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")

        self.north_m += velocity[0] * dt
        self.east_m += velocity[1] * dt
        self.down_m += velocity[2] * dt
