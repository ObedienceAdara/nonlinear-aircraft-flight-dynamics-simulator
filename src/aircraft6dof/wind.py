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
    """Three-axis Dryden-spectrum shaping filter.

    The longitudinal channel uses the first-order Dryden filter shape;
    lateral and vertical channels use the second-order shape with a
    sqrt(3) numerator zero. Exact discrete-time state transitions and their
    integrated process-noise covariance are used for each fixed-airspeed step.
    The configured sigma values define the stationary RMS of each output.

    sigma_m_s and scale_length_m are ordered [longitudinal, lateral, vertical]
    in the caller's turbulence axes. Rotate the returned vector into NED before
    assigning it to Environment.gust_ned_m_s if those axes are not already NED.

    This is a configurable Dryden spectral filter, not a complete MIL-F-8785C
    implementation: it does not select severity/altitude parameters, model
    angular turbulence, or claim certification-level conformance.
    """

    sigma_m_s: np.ndarray
    scale_length_m: np.ndarray
    seed: int = 1

    def __post_init__(self):
        self.sigma_m_s = np.asarray(self.sigma_m_s, dtype=float)
        self.scale_length_m = np.asarray(self.scale_length_m, dtype=float)
        if self.sigma_m_s.shape != (3,) or self.scale_length_m.shape != (3,):
            raise ValueError("sigma and scale_length must be (3,)")
        if not np.all(np.isfinite(self.sigma_m_s)) or np.any(self.sigma_m_s < 0.0):
            raise ValueError("sigma must be finite and non-negative")
        if not np.all(np.isfinite(self.scale_length_m)) or np.any(self.scale_length_m <= 0.0):
            raise ValueError("scale_length must be finite and positive")
        self.state_m_s = np.zeros(3)
        # For each transverse axis, store the two states of the repeated-pole
        # realization: x1'=-a*x1+noise, x2'=x1-a*x2.
        self._transverse_states = np.zeros((2, 2))
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _noise_integrals(rate: float, dt: float) -> tuple[float, float, float]:
        """Integrals of exp(-2*a*t) * [1, t, t^2] over [0, dt].

        A small-argument series avoids cancellation when a*dt is close to zero.
        """
        x = 2.0 * rate * dt
        if x < 1e-3:
            x2, x3, x4 = x*x, x*x*x, x*x*x*x
            i0 = dt * (1.0 - x/2.0 + x2/6.0 - x3/24.0 + x4/120.0)
            i1 = dt**2 * (0.5 - x/3.0 + x2/8.0 - x3/30.0 + x4/144.0)
            i2 = dt**3 * (1.0/3.0 - x/4.0 + x2/10.0 - x3/36.0 + x4/168.0)
            return i0, i1, i2

        decay = np.exp(-x)
        i0 = -np.expm1(-x) / (2.0 * rate)
        i1 = (-np.expm1(-x) - x*decay) / (4.0 * rate**2)
        i2 = (-np.expm1(-x) - decay*(x + 0.5*x*x)) / (4.0 * rate**3)
        return i0, i1, i2

    def step(self, airspeed_m_s: float, dt: float) -> np.ndarray:
        """Advance all three Dryden channels by one time step.

        Airspeed is held constant over this step. State transitions and
        process-noise covariance are exact for that local condition, rather
        than Euler-discretizing the filter differential equations.
        """
        V_raw = float(airspeed_m_s)
        dt = float(dt)
        if not np.isfinite(V_raw):
            raise ValueError("airspeed_m_s must be finite")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0")
        V = max(V_raw, 1.0)
        rates = V / self.scale_length_m
        decay = np.exp(-rates * dt)

        # First-order longitudinal channel; its stationary standard deviation
        # equals sigma[0].
        innovation_std = self.sigma_m_s[0] * np.sqrt(max(0.0, -np.expm1(-2.0*rates[0]*dt)))
        self.state_m_s[0] = decay[0]*self.state_m_s[0] + innovation_std*self.rng.standard_normal()

        # Second-order Dryden transfer-function shape for lateral and vertical
        # turbulence. A two-state realization is discretized exactly for the
        # current airspeed. Process-noise intensity pi and the output gain make
        # the stationary output RMS equal to the configured sigma.
        for axis in (1, 2):
            rate = float(rates[axis])
            i0, i1, i2 = self._noise_integrals(rate, dt)
            q00, q01, q11 = np.pi*i0, np.pi*i1, np.pi*i2
            l00 = np.sqrt(max(q00, 0.0))
            l10 = q01/l00 if l00 > 0.0 else 0.0
            l11 = np.sqrt(max(q11 - l10*l10, 0.0))
            z0, z1 = self.rng.standard_normal(2)
            noise0 = l00*z0
            noise1 = l10*z0 + l11*z1

            old0, old1 = self._transverse_states[axis-1]
            new0 = decay[axis]*old0 + noise0
            new1 = decay[axis]*(old1 + dt*old0) + noise1
            self._transverse_states[axis-1] = (new0, new1)

            # H(s) has a double pole at -rate and a zero at -rate/sqrt(3):
            # y is proportional to (s + rate/sqrt(3))/(s + rate)^2.
            shape_gain = self.sigma_m_s[axis]*np.sqrt(3.0*rate/np.pi)
            self.state_m_s[axis] = shape_gain*(new0 + (rate/np.sqrt(3.0) - rate)*new1)

        return self.state_m_s.copy()
