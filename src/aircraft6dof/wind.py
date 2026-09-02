"""Wind, deterministic gust, and stochastic turbulence models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(init=False)
class OneMinusCosineGust:
    """Finite-duration one-minus-cosine gust in NED coordinates.

    Preferred API
    -------------
    ``OneMinusCosineGust(amplitude_m_s, start_s, duration_s)``

    ``amplitude_m_s`` is a 3-vector containing the signed peak gust velocity
    in NED axes. The gust ramps from zero to the requested vector amplitude
    over the first half of ``duration_s`` and returns to zero over the second
    half.

    Backward-compatible API
    -----------------------
    The original simulator prototype accepted
    ``(magnitude_m_s, direction_ned, start_s, rise_s, hold_s, fall_s)``.
    That form is retained so existing callers are not broken while the new
    vector API is used by the project-level runner.
    """

    amplitude_m_s: np.ndarray
    start_s: float
    duration_s: float

    # Legacy timing parameters are retained for introspection/compatibility.
    rise_s: float
    hold_s: float
    fall_s: float

    def __init__(self, *args: Any, amplitude_m_s=None, start_s=None, duration_s=None,
                 magnitude_m_s=None, direction_ned=None, rise_s=None,
                 hold_s=0.0, fall_s=None):
        # Six positional arguments are the legacy constructor.
        if args:
            if len(args) == 6:
                if any(v is not None for v in (amplitude_m_s, start_s, duration_s,
                                               magnitude_m_s, direction_ned,
                                               rise_s, fall_s)):
                    raise TypeError("Do not mix positional and keyword gust constructor forms")
                magnitude_m_s, direction_ned, start_s, rise_s, hold_s, fall_s = args
            elif len(args) == 3:
                if any(v is not None for v in (amplitude_m_s, start_s, duration_s,
                                               magnitude_m_s, direction_ned,
                                               rise_s, fall_s)):
                    raise TypeError("Do not mix positional and keyword gust constructor forms")
                amplitude_m_s, start_s, duration_s = args
            else:
                raise TypeError(
                    "OneMinusCosineGust expects either 3 positional arguments "
                    "(amplitude_m_s, start_s, duration_s) or the legacy 6-argument form"
                )

        # Legacy keyword form.
        if amplitude_m_s is None and magnitude_m_s is not None:
            if direction_ned is None or start_s is None or rise_s is None or fall_s is None:
                raise TypeError(
                    "Legacy gust form requires magnitude_m_s, direction_ned, start_s, rise_s, and fall_s"
                )
            direction = np.asarray(direction_ned, dtype=float)
            if direction.shape != (3,):
                raise ValueError("direction_ned must have shape (3,)")
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                amplitude = np.zeros(3)
            else:
                amplitude = float(magnitude_m_s) * direction / norm
            duration = float(rise_s) + float(hold_s) + float(fall_s)
            self.amplitude_m_s = amplitude
            self.start_s = float(start_s)
            self.duration_s = duration
            self.rise_s = float(rise_s)
            self.hold_s = float(hold_s)
            self.fall_s = float(fall_s)
            self._legacy_profile = True
            return

        if amplitude_m_s is None or start_s is None or duration_s is None:
            raise TypeError("Gust requires amplitude_m_s, start_s, and duration_s")

        amplitude = np.asarray(amplitude_m_s, dtype=float)
        if amplitude.shape != (3,):
            raise ValueError("amplitude_m_s must have shape (3,)")
        if not np.all(np.isfinite(amplitude)):
            raise ValueError("amplitude_m_s must contain finite values")
        start = float(start_s)
        duration = float(duration_s)
        if not np.isfinite(start):
            raise ValueError("start_s must be finite")
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be finite and > 0")

        self.amplitude_m_s = amplitude
        self.start_s = start
        self.duration_s = duration
        self.rise_s = 0.5 * duration
        self.hold_s = 0.0
        self.fall_s = 0.5 * duration
        self._legacy_profile = False

    def value(self, t: float) -> np.ndarray:
        """Return gust velocity vector at time ``t`` in seconds."""
        tau = float(t) - self.start_s
        if tau <= 0.0 or tau >= self.duration_s:
            return np.zeros(3)

        if getattr(self, "_legacy_profile", False):
            if tau <= self.rise_s:
                phase = 0.5 * (1.0 - np.cos(np.pi * tau / self.rise_s))
            elif tau <= self.rise_s + self.hold_s:
                phase = 1.0
            else:
                s = tau - self.rise_s - self.hold_s
                phase = 0.5 * (1.0 + np.cos(np.pi * s / self.fall_s))
        else:
            half = 0.5 * self.duration_s
            if tau <= half:
                phase = 0.5 * (1.0 - np.cos(np.pi * tau / half))
            else:
                s = tau - half
                phase = 0.5 * (1.0 + np.cos(np.pi * s / half))

        return phase * self.amplitude_m_s


@dataclass
class DrydenTurbulence:
    """Reproducible correlated turbulence approximation.

    This is intentionally documented as a Dryden-style stochastic process,
    not as a certification-grade MIL-F-8785 implementation.
    """

    sigma_m_s: np.ndarray
    scale_length_m: np.ndarray
    seed: int = 1

    def __post_init__(self):
        self.sigma_m_s = np.asarray(self.sigma_m_s, dtype=float)
        self.scale_length_m = np.asarray(self.scale_length_m, dtype=float)
        if self.sigma_m_s.shape != (3,) or self.scale_length_m.shape != (3,):
            raise ValueError("sigma and scale_length must be (3,)")
        if np.any(self.sigma_m_s < 0.0) or np.any(self.scale_length_m <= 0.0):
            raise ValueError("sigma must be non-negative and scale_length must be positive")
        self.state_m_s = np.zeros(3)
        self.rng = np.random.default_rng(self.seed)

    def step(self, airspeed_m_s: float, dt: float) -> np.ndarray:
        """Advance the correlated stochastic process by one time step."""
        V = max(float(airspeed_m_s), 1.0)
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        tau = np.maximum(self.scale_length_m / V, 1e-3)
        a = np.exp(-dt / tau)
        self.state_m_s = (
            a * self.state_m_s
            + self.sigma_m_s * np.sqrt(np.maximum(0.0, 1.0 - a * a))
            * self.rng.standard_normal(3)
        )
        return self.state_m_s.copy()
