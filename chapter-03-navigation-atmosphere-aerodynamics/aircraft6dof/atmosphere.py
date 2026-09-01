"""Compact standard-atmosphere model."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AtmosphereState:
    altitude_m: float
    temperature_K: float
    pressure_Pa: float
    density_kg_m3: float
    speed_of_sound_m_s: float


T0 = 288.15
P0 = 101325.0
R = 287.05287
G0 = 9.80665
GAMMA = 1.4
LAPSE = -0.0065
T11 = 216.65
H11 = 11000.0


def standard_atmosphere(altitude_m: float) -> AtmosphereState:
    """Return temperature, pressure, density and sound speed up to 20 km.

    Uses the standard linear-lapse troposphere through 11 km and an
    isothermal layer above 11 km. Altitude is geometric height above sea level.
    """
    h = float(altitude_m)
    if not math.isfinite(h):
        raise ValueError("altitude_m must be finite.")
    if h < 0.0 or h > 20000.0:
        raise ValueError("altitude_m must be between 0 and 20,000 m.")

    if h <= H11:
        T = T0 + LAPSE * h
        exponent = -G0 / (LAPSE * R)
        P = P0 * (T / T0) ** exponent
    else:
        T = T11
        P11 = P0 * (T11 / T0) ** (-G0 / (LAPSE * R))
        P = P11 * math.exp(-G0 * (h - H11) / (R * T11))

    rho = P / (R * T)
    a = math.sqrt(GAMMA * R * T)

    return AtmosphereState(h, T, P, rho, a)
